import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os, sys, time
import numpy as np


sys.path.append('/kaggle/input/finetune')
from utils import RecorderMeter
import resnet

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
cudnn.benchmark = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:",device)
best_acc = 0  # best test accuracy

# hyperparams
epochs=150 # total epochs
###

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

net = resnet.resnet32()
net = net.to(device)

print('loading checkpoint')
checkpoint = torch.load('../input/finetune/pruned.pth')
net.load_state_dict(checkpoint['net'])
pruned_idx=checkpoint['pruned']
print(pruned_idx)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75])
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
recorder = RecorderMeter(epochs)



def zeroize():
    for i, weights in enumerate(net.parameters()):
        if i%3 == 0 and i<91:
            layer = i//3
            for filter_idx in pruned_idx[layer]:
                weights.data[filter_idx] = 0
                
# Training
def train(epoch):
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

    net.eval()
    print("\nTrain Accuracy:",acc, "\n Train Loss:", train_loss)
    return acc, train_loss

 
def test(epoch):
    global best_acc
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

    if acc > best_acc:
        print('Saving best checkpoint..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'pruned_idx': pruned_idx,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/final.pth')
        best_acc = acc
    
    print("\nTest Accuracy:",acc, "\n Test Loss:", test_loss)
    return acc, test_loss


if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
start_time = time.time()

test(0)
for epoch in range(0, epochs):

    print("Training epoch",epoch)
    train_acc, train_loss = train(epoch)

    print("Testing epoch",epoch)
    test_acc, test_loss = test(epoch)

    scheduler.step()
    recorder.update(epoch, train_loss, train_acc, test_loss, test_acc)
    recorder.plot_curve_acc('finetune_curve.png')
    recorder.plot_curve_loss('finetune_loss.png')


    epoch_time = time.time() - start_time
    print("Epoch duration",epoch_time/60,"mins")
    start_time = time.time()

print("Best acc:",best_acc)

