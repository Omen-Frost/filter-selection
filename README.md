# filter-selection

Pruning Resnet32 on Cifar10 dataset.<br/><br/>

Use train.py for training without pruning. Model output: resnet32_baseline.pth <br/>
Use main.py for training with pruning. Model output: pruned.pth <br/>
use fine_tune.py for fine-tuning of pruned.pth. Model output resnet32_pruned.pth. <br/>


Requirements: <br/>
  Python 3.8, pytorch 1.10.2, torchvision 0.11.3
