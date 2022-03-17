import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
from scipy.spatial import distance
import numpy as np
import random

sys.path.append('/kaggle/input/lrfimport')
from utils import RecorderMeter
import resnet

#####
# Hyperparams
warmup = 15  # initial training
resume = True  # resume from checkpoint
pruning_ratio = 0.5  # percentage of filters to remove
model = 'resnet32' # 'resnet32' or 'resnet56'
load_path = 'pretrained/model_32.th' # Original trained model path
#####

# speedup
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
cudnn.benchmark = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device)

# Data
print('==> Preparing data..')
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)


# Model
print('==> Building model..')

if model == 'resnet32':
    net = resnet.resnet32()
elif model == 'resnet56':
    net = resnet.resnet56()

net = net.to(device)

# for name, p in net.named_parameters():
    # if len(p.size())==4 and p.size(3)>1:
    # print(name,p.size())
# see layer shapes of the network
# for index, item in enumerate(net.parameters()):
#     print(index,item.shape)
# exit()


criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.001,
                      momentum=0.9, weight_decay=1e-4)

if model == 'resnet32':
    recorder = RecorderMeter(157)
elif model == 'resnet56':
    recorder = RecorderMeter(142)


features = {}  # stores output feature maps for each layer
pruned_idx = {}  # stores indices of filters that are pruned for each layer
layer_begin = 0
layer_interval = 3  # interval of conv2d layers
N_layers = 0  # number of conv2d layers in the network
ckp_interval = 5
shape_layers = []

if resume:
    print('loading checkpoint')
    checkpoint = torch.load(load_path)
    device_ids = [0]
    net = torch.nn.DataParallel(net, device_ids=device_ids)
    net.load_state_dict(checkpoint['state_dict'])


# Hook for extracting features from intermediate layers
def get_features(name):
    def hook(model, input, output):
        # capture original output fmaps during forward pass in pruning
        features[name] = output.detach().cpu().numpy()

    return hook


def zeroize():
    for i, weights in enumerate(net.parameters()):
        if i%3 == 0 and i//3<N_layers:
            layer = i//3
            for filter_idx in pruned_idx[layer]:
                weights.data[filter_idx] = 0

def prune(layer):
    L=[]
    for x in range(0,shape_layers[layer][0]):
        if x not in pruned_idx[layer]:
            L.append(x)
    pruned_idx[layer].append(random.sample(L,1)[0])

# Training
def train():
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        zeroize()
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

    print("\nTrain Accuracy:", acc, "\n Train Loss:", train_loss)
    net.eval()
    return acc, train_loss


def test():
    net.eval()
    zeroize()
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

    print("\nTest Accuracy:", acc, "\n Test Loss:", test_loss)
    return acc, test_loss

def filterStatus():
    for i, weights in enumerate(net.parameters()):
        if i%3 == 0 and i//3<N_layers:
            layer = i//3
            print("\n\nLayer:",layer, weights.data.shape)
            L = []
            for idx, data in enumerate(weights.data):
                if data.sum() == 0 :
                    L.append(idx)
            if L:
                print("Pruned",len(L),"filters:",L)

layer_id = 0
for layer in net.modules():
    if isinstance(layer, torch.nn.modules.conv.Conv2d):
        pruned_idx[layer_id]=[]
        layer_id += 1
        shape_layers.append(layer.weight.shape)
N_layers = layer_id
print(shape_layers[1:])

if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')


print("Warmup Stage>")
start_time = time.time()
for epoch in range(0,warmup):

    print("Training epoch", epoch)
    train_acc, train_loss = train()

    print("Testing epoch", epoch)
    test_acc, test_loss = test()

    # scheduler.step()
    recorder.update(epoch, train_loss, train_acc, test_loss,
                    test_acc)
    recorder.plot_curve_acc('prune_curve.png')
    recorder.plot_curve_loss('prune_curve.png')

    epoch_time = time.time() - start_time
    print("Epoch duration", epoch_time/60, "mins")
    start_time = time.time()

i=0
j=4
if model == 'resnet32':
    j=5
elif model == 'resnet56':
    j=10

print("Pruning>")
for layer in reversed(range(0,N_layers)):

    print(pruned_idx)
    print("#### Pruning layer",layer,"####")
    pruning_num = int(round(shape_layers[layer][0] * pruning_ratio))
    pruning_num += pruning_num/4
    for _ in range(pruning_num):
        start_time = time.time()

        print("Pruning one filter")
        prune(layer)
        i+=1

        print("Fine tuning")
        train_acc, train_loss = train()

        print("Testing")
        test_acc, test_loss = test()

        epoch_time = time.time() - start_time
        print("single prune duration", epoch_time/60, "mins")
               
        if i%j ==0:
            recorder.update(warmup-1+i//j, train_loss, train_acc, test_loss,
                        test_acc)
            recorder.plot_curve_acc('prune_curve.png')
            recorder.plot_curve_loss('prune_curve.png')
            
print(i)
print(pruned_idx)

test_acc, test_loss = test()
print("TEST ACC",test_acc, "Loss",test_loss)

print('Saving pruned model..')
state = {
    'net': net.state_dict(),
    'pruned': pruned_idx,
}
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
torch.save(state, './checkpoint/pruned.pth')
