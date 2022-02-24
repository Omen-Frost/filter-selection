import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os, sys, time
from scipy.spatial import distance
import math
import numpy as np
import cvxpy as cp

# sys.path.append('/kaggle/input/import')
from utils import RecorderMeter
import resnets


# torch.autograd.set_detect_anomaly(False)
# torch.autograd.profiler.profile(False)
# torch.autograd.profiler.emit_nvtx(False)
# cudnn.benchmark = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:",device)
best_acc = 0  # best test accuracy

###
# hyperparams
epochs=150 # total epochs
prune_interval = 1
start_epoch=0
resume=False # resume from checkpoint
layer_end = 91 # for resnet32
pruning_ratio = 0.5 # percentage of filters to keep
inc_batch_sz = 64 # incoming filter batch size
# ext_max_size = 128
###

# Data
print('==> Preparing data..')
normalize = transforms.Normalize(mean=[x / 255 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255 for x in [63.0, 62.1, 66.7]]) # from FPGM
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

# split testset into test and valid set
split = int(0.5*len(testset))
validset = torch.utils.data.Subset(testset, range(split))
testset = torch.utils.data.Subset(testset, range(split, len(testset)))

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)
validloader = torch.utils.data.DataLoader(
    validset, batch_size=100, shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# Model
print('==> Building model..')

net = resnets.resnet32()
net = net.to(device)

# see layer shapes of the network
# for index, item in enumerate(net.parameters()):
#     print(index,item.shape)
# # #     print(item.data)
# exit()
# for name,module in net.named_children():
#     print(name)


criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100], last_epoch=start_epoch - 1)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
recorder = RecorderMeter(epochs)

features = {} # stores output feature maps for each layer
pruned_idx = {} # stores indices of filters that are pruned for each layer
layer_begin=0
layer_interval = 3
N_layers = 0
ckp_interval=5
shape_layers = []
is_pruning = False

if resume:
    print('loading checkpoint')
    checkpoint = torch.load('../input/checkpoint/ckp.pth')
    recorder = checkpoint['recorder']
    start_epoch=checkpoint['epoch']+1
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # scheduler.load_state_dict(checkpoint['scheduler'])
    pruned_idx=checkpoint['pruned_idx']


def solver1(incD):
    z = cp.Variable((incD.shape[0],incD.shape[1]), boolean=True)  
    constraints = [ cp.sum(z, axis=1) == 1, cp.sum(cp.max(z,axis=0)) <= math.ceil(incD.shape[0]*pruning_ratio)]
    objective    = cp.Minimize(cp.sum(cp.multiply(incD,z)))
    prob = cp.Problem(objective, constraints)
    prob.solve()  
    # print("status:", prob.status)
    # print("optimal value", prob.value)

    L = np.max(z.value,axis=0).astype(int).tolist()
    selected_idx = []
    for i,b in enumerate(L):
        if b:
            selected_idx.append(i)
    return selected_idx

def solver2(incD, incXextD):
    Zn = cp.Variable((incD.shape[0],incD.shape[1]), boolean=True)  
    Zo = cp.Variable((incXextD.shape[0],incXextD.shape[1]), boolean=True)  
    constraints = [ cp.sum(Zn, axis=1) + cp.sum(Zo, axis=1) == 1, cp.sum(cp.max(Zn,axis=0)) <= math.ceil(incD.shape[0]*pruning_ratio)]
    objective    = cp.Minimize(cp.sum(cp.multiply(incD,Zn)) + cp.sum(cp.multiply(incXextD,Zo)))
    prob = cp.Problem(objective, constraints)
    prob.solve()  

    L = np.max(Zn.value,axis=0).astype(int).tolist()
    selected_idx = []
    for i,b in enumerate(L):
        if b:
            selected_idx.append(i)
    return selected_idx


# finds optimal subset of filters from a single layer
def findFilterSubset(D):
    if D.shape[0] <= inc_batch_sz :
        return solver1(D)
    
    batches = D.shape[0]//inc_batch_sz
    ext_set = []
    for i in range(0,batches):
        if i==0:
            ext_set+= solver1(D[0:inc_batch_sz, 0:inc_batch_sz])
        else :
            I = solver2(D[inc_batch_sz*i:inc_batch_sz*(i+1), inc_batch_sz*i:inc_batch_sz*(i+1)], 
                    D[np.ix_(list(range(inc_batch_sz*i,inc_batch_sz*(i+1))), ext_set)] )
            for j in I:
                ext_set.append(j+i*inc_batch_sz)

    return ext_set



# zeroizes the gradients for pruned filters
# def do_zero_grad(net):
#     if not pruned_idx:
#         return
#     for index, item in enumerate(net.parameters()):
#         if index%3 == 0 and index<layer_end:
#             layer = index//3
#             for filter_idx in pruned_idx[layer]:
#                 item.grad.data[filter_idx]=0

# Pruning 
def prune():
    net.train()
    global is_pruning
    is_pruning=True
    dist_mat = [] # stores output fmaps' pairwise distances
    for layer in range(0,N_layers):
        dist_mat.append(np.zeros([shape_layers[layer][0], shape_layers[layer][0] ]))

    valid_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)   
            valid_loss += loss.item()*inputs.size(0)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            for layer in range(0,N_layers):
            # features[layer]: B * F * H * W 
                if device=="cpu":
                    for outputs in features[layer]:
                        vec = outputs.reshape(outputs.shape[0], -1)
                        dist_mat[layer]+= distance.cdist(vec,vec,"euclidean")
                else:
                    tns = features[layer].flatten(2)
                    dist_mat[layer]+= torch.cdist(tns,tns,p=2).sum(0).cpu().detach().numpy()
            

        # average distances between fmaps over whole set
        dist_mat[:] = [mat/total for mat in dist_mat]
        
        for layer in range(0,N_layers):
            selected = findFilterSubset(dist_mat[layer])
            L=[]
            for idx in range(0, shape_layers[layer][0]):
                if not idx in selected:
                    L.append(idx)
            pruned_idx[layer]=L
    
    is_pruning=False

    acc = 100.*correct/total
    valid_loss = valid_loss/total
    
    print("\nValid Accuracy:",acc, "\n Valid Loss:", valid_loss)
    return acc, valid_loss
        

# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()*inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        

    acc = 100.*correct/total
    train_loss = train_loss/total

    print("\nTrain Accuracy:",acc, "\n Train Loss:", train_loss)
    return acc, train_loss

 
def test(epoch):
    global best_acc
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)   
            test_loss += loss.item()*inputs.size(0)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            

    acc = 100.*correct/total
    test_loss = test_loss/total

    if acc > best_acc:
        print('Saving best checkpoint..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'pruned': pruned_idx,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/best.pth')
        best_acc = acc
    
    print("\nTest Accuracy:",acc, "\n Test Loss:", test_loss)
    return acc, test_loss


# hook for extracting features from intermediate layers
def get_features(name):
    def hook(model, input, output):
        if is_pruning:
            # capture output fmaps during forward pass in pruning
            features[name] = output.detach()
        elif pruned_idx:
            # zeroize outputs of pruned filters for forward pass in train
            for idx in pruned_idx[name]:
                output[:,idx]=0.0 
    return hook

# register forward hooks for each conv layer
layer_id=0
for layer in net.modules():
    if isinstance(layer, torch.nn.modules.conv.Conv2d):
        layer.register_forward_hook(get_features(layer_id))
        layer_id+=1
        shape_layers.append(layer.weight.shape)
N_layers = layer_id

if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
start_time = time.time()

for epoch in range(start_epoch, epochs):

    print("Training epoch",epoch)
    train_acc, train_loss = train(epoch)

    if (epoch+1)%prune_interval==0:
        print("Pruning epoch",epoch)
        valid_acc, valid_loss = prune()

    print("Testing epoch",epoch)
    test_acc, test_loss = test(epoch)

    # scheduler.step()
    recorder.update(epoch, train_loss, train_acc, test_loss, test_acc, valid_loss, valid_acc)
    recorder.plot_curve_acc('prune_curve.png')
    recorder.plot_curve_loss('prune_curve.png')

    if epoch%ckp_interval == 0 :
        # Save checkpoint.
        print('Saving checkpoint..')
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
            'recorder': recorder,
            'optimizer': optimizer.state_dict(),
            # 'scheduler': scheduler.state_dict(),
            'pruned_idx': pruned_idx,
        }
        torch.save(state, './checkpoint/ckp.pth')

    epoch_time = time.time() - start_time
    print("Epoch duration",epoch_time/60,"mins")
    start_time = time.time()

print(pruned_idx)
print("Best acc:",best_acc)
