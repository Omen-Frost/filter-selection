import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os, argparse, time
from scipy.spatial import distance
import numpy as np
import cvxpy as cp

from arch import resnet
from utils import RecorderMeter

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:",device)
cudnn.benchmark = True
best_acc = 0  # best test accuracy
epochs=1 # total epochs
pruning_ratio = 0.5 # percentage of filters to keep

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# Model
print('==> Building model..')
net = resnet.ResNet18()
net = net.to(device)

# see layer shapes of the network
# for index, item in enumerate(net.parameters()):
#     print(index,item.shape)
#     print(item.data)
# exit()
# for name,module in net.named_children():
#     print(name)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
recorder = RecorderMeter(epochs)

features = {}
layer_begin=0
layer_end = 60 # for resnet18
layer_interval = 3
N_layers = 0
shape_layers = []

def findFilterSubset(D):
    z = cp.Variable((D.shape[0],D.shape[1]), boolean=True)  
    constraints = [ cp.sum(z, axis=1) == 1, cp.sum(cp.max(z,axis=0)) <= D.shape[0]*pruning_ratio]
    objective    = cp.Minimize(cp.sum(cp.multiply(D,z)))
    prob = cp.Problem(objective, constraints)
    prob.solve()  
    print("status:", prob.status)
    print("optimal value", prob.value)

    selected_idx = np.max(z.value,axis=0)
    prune_idx = 1 - selected_idx.astype(int)
    return prune_idx
   
def zeroize(net, prune_idx):

    for i, weights in enumerate(net.parameters()):
        if i%3 == 0 and i<layer_end:
            layer = i//3
            print("zeroizing", layer)
            for filter_idx, prune in enumerate(prune_idx[layer]):
                if prune:
                    weights.data[filter_idx] = 0


# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    dist_mat = [] # stores output fmaps' pairwise distances
    for layer in range(0,N_layers):
        dist_mat.append(np.zeros([shape_layers[layer][0], shape_layers[layer][0] ]))

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # print(len(trainloader), inputs.size(), targets.size(), sep='\n\n\n')
        
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

        for layer in range(0,N_layers):
            for outputs in features[layer]:
                vec = outputs.reshape(outputs.shape[0], -1)
                dist_mat[layer]+= distance.cdist(vec,vec,"euclidean")
        break

    # average distances between fmaps over whole training set
    for mat in dist_mat:
        mat = mat/trainset.__len__()

    prune_idx = []
    for layer in range(0,N_layers):
        # filter selection
        prune_idx.append(findFilterSubset(dist_mat[layer]))
        
    # prune selected filters by zeroizing
    zeroize(net, prune_idx)


    acc = 100.*correct/total
    train_loss = 100.*train_loss/total

    print("\nTrain Accuracy:",acc, "\n Train Loss:", train_loss)
    return acc, train_loss

 
def test(epoch):
    global best_acc
    net.eval()
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
    test_loss = 100.*test_loss/total
    # Save checkpoint.
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    
    print("\nTest Accuracy:",acc, "\n Test Loss:", test_loss)
    return acc, test_loss


# hook for extracting features from intermediate layers
def get_features(name):
    def hook(model, input, output):
        features[name] = output.cpu().detach().numpy()
    return hook

# register forward hooks for each conv layer
layer_id=0
for layer in net.modules():
    if isinstance(layer, torch.nn.modules.conv.Conv2d):
        layer.register_forward_hook(get_features(layer_id))
        layer_id+=1
        shape_layers.append(layer.weight.shape)
        # print(layer)
N_layers = layer_id

start_time = time.time()

for epoch in range(0, epochs):
    print("Training epoch",epoch)
    train_acc, train_loss = train(epoch)

    print("Testing epoch",epoch)
    test_acc, test_loss = test(epoch)

    scheduler.step()
    recorder.update(epoch, train_loss, train_acc, test_loss, test_acc)
    recorder.plot_curve('curve.png')

    epoch_time = time.time() - start_time
    print("Epoch duration",epoch_time/60,"mins")
    start_time = time.time()


