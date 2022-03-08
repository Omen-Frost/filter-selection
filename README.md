# filter-selection

Pruning Resnet32 on Cifar10 dataset.

Use train.py for training without pruning. Model output: resnet32_baseline.pth
Use main.py for training with pruning. Model output: pruned.pth
use fine_tune.py for fine-tuning of pruned.pth. Model output resnet32_pruned.pth.


Requirements: 
  Python 3.8, pytorch 1.10.2, torchvision 0.11.3
