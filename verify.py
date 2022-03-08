from tokenize import blank_re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os, argparse, time
from utils import RecorderMeter

import resnet


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

# torch.autograd.set_detect_anomaly(False)
# torch.autograd.profiler.profile(False)
# torch.autograd.profiler.emit_nvtx(False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:",device)
# cudnn.benchmark = True

layer_end=91

# Data
print('==> Preparing data..')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
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

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = resnet.resnet32() 
net = net.to(device)

criterion = nn.CrossEntropyLoss()


print('loading best checkpoint')
checkpoint = torch.load('final.pth', map_location=torch.device('cpu'))
# epoch=checkpoint['epoch']
net.load_state_dict(checkpoint['net'])
# acc = checkpoint['acc']
# pruned_idx=checkpoint['pruned']
# print("epochs:",epoch,"\nacc:",acc)
# print(pruned_idx)
 
def test():
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
    test_loss = test_loss/total

    
    print("\nTest Accuracy:",acc, "\n Test Loss:", test_loss)
    return acc, test_loss

def filterStatus():
    for i, weights in enumerate(net.parameters()):
        if i%3 == 0 and i<layer_end:
            layer = i//3
            print("\n\nLayer:",layer, weights.data.shape)
            L = []
            for idx, data in enumerate(weights.data):
                if data.numpy().sum() == 0 :
                    L.append(idx)
            if L:
                print("Pruned",len(L),"filters:",L)


# def zeroize(net):

#     for i, weights in enumerate(net.parameters()):
#         if i%3 == 0 and i<layer_end:
#             layer = i//3

#             for filter_idx in pruned_idx[layer]:
#                 weights.data[filter_idx] = 0

# zeroize(net)
filterStatus()
# test()