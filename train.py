import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os, argparse, time

from utils import RecorderMeter
import resnets


# torch.autograd.set_detect_anomaly(False)
# torch.autograd.profiler.profile(False)
# torch.autograd.profiler.emit_nvtx(False)
# cudnn.benchmark = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:",device)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
epochs = 150
resume=0

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
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = resnets.resnet32()
net = net.to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100], last_epoch=start_epoch - 1)

recorder = RecorderMeter(epochs)

if resume:
    print('loading checkpoint')
    checkpoint = torch.load('./checkpoint/base_ckp.pth')
    recorder = checkpoint['recorder']
    start_epoch=checkpoint['epoch']+1
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

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

    # Save checkpoint.
    if acc > best_acc:
        print('Saving Best..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/best_base.pth')
        best_acc = acc
    
    print("\nTest Accuracy:",acc, "\n Test Loss:", test_loss)
    return acc, test_loss


start_time = time.time()

for epoch in range(start_epoch, epochs): #Run till convergence
    print("Training epoch",epoch)
    train_acc, train_loss = train(epoch)

    print("Testing epoch",epoch)
    test_acc, test_loss = test(epoch)

    scheduler.step()
    recorder.update(epoch, train_loss, train_acc, test_loss, test_acc)
    recorder.plot_curve_acc('train_curve.png')
    recorder.plot_curve_loss('train_curve.png')

    # Save checkpoint.
    print('Saving checkpoint..')
    state = {
        'net': net.state_dict(),
        'epoch': epoch,
        'recorder': recorder,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/base_ckp.pth')

    epoch_time = time.time() - start_time
    print("Epoch duration",epoch_time/60,"mins")
    start_time = time.time()

print("Best acc",best_acc)