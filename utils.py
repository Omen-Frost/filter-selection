import os, sys, time
import numpy as np
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


class RecorderMeter(object):
  """Computes and stores the minimum loss value and its epoch index"""
  def __init__(self):
    self.limit=305
    self.reset()
    self.idx=0

  def reset(self):
    self.epoch_losses  = np.zeros((self.limit, 3), dtype=np.float32) # [epoch, train/val]
    self.epoch_losses  = self.epoch_losses - 1

    self.epoch_accuracy= np.zeros((self.limit, 3), dtype=np.float32) # [epoch, train/val]
    self.epoch_accuracy= self.epoch_accuracy

  def update(self, train_loss, train_acc, test_loss, test_acc, val_loss=0.0, val_acc=0.0):
    self.epoch_losses  [self.idx, 0] = train_loss
    self.epoch_losses  [self.idx, 1] = test_loss
    self.epoch_losses  [self.idx, 2] = val_loss
    self.epoch_accuracy[self.idx, 0] = train_acc
    self.epoch_accuracy[self.idx, 1] = test_acc
    self.epoch_accuracy[self.idx, 2] = val_acc
    self.idx += 1
  
  def plot_curve_acc(self, save_path):
    title = 'accuracy vs epochs'
    dpi = 80  
    width, height = 1200, 800
    legend_fontsize = 10
    scale_distance = 48.8
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    x_axis = np.array([i for i in range(self.idx)]) # epochs
    y_axis = np.zeros(self.idx)

    plt.xlim(0, self.idx)
    plt.ylim(0, 100)
    interval_y = 5
    interval_x = 10
    plt.xticks(np.arange(0, self.idx + interval_x, interval_x))
    plt.yticks(np.arange(0, 100 + interval_y, interval_y))
    plt.grid()
    plt.title(title, fontsize=20)
    plt.xlabel('the training epoch', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)
  
    y_axis[:] = self.epoch_accuracy[:self.idx, 0]
    plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_accuracy[:self.idx, 1]
    plt.plot(x_axis, y_axis, color='y', linestyle='-', label='test-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    if self.epoch_accuracy[:, 2].any():
      y_axis[:] = self.epoch_accuracy[:self.idxself.idx, 2]
      plt.plot(x_axis, y_axis, color='r', linestyle='-', label='valid-accuracy', lw=2)
      plt.legend(loc=4, fontsize=legend_fontsize)

    if save_path is not None:
      fig.savefig('acc_'+save_path, dpi=dpi, bbox_inches='tight')
      print ('---- save figure {} into {}'.format(title, 'acc_'+save_path))
    plt.close(fig)

  def plot_curve_loss(self, save_path):
      title = 'loss vs epochs'
      dpi = 80  
      width, height = 1200, 800
      legend_fontsize = 10
      scale_distance = 48.8
      figsize = width / float(dpi), height / float(dpi)

      fig = plt.figure(figsize=figsize)
      x_axis = np.array([i for i in range(self.idx)]) # epochs
      y_axis = np.zeros(self.idx)

      plt.xlim(0, self.idx)
      plt.ylim(0, 100)
      interval_y = 5
      interval_x = 10
      plt.xticks(np.arange(0, self.idx + interval_x, interval_x))
      plt.yticks(np.arange(0, 100 + interval_y, interval_y))
      plt.grid()
      plt.title(title, fontsize=20)
      plt.xlabel('the training epoch', fontsize=16)
      plt.ylabel('losses', fontsize=16)
    
      y_axis[:] = self.epoch_losses[:self.idx, 0]
      plt.plot(x_axis, y_axis*40, color='g', linestyle=':', label='train-loss-x40', lw=2)
      plt.legend(loc=4, fontsize=legend_fontsize)

      y_axis[:] = self.epoch_losses[:self.idx, 1]
      plt.plot(x_axis, y_axis*40, color='y', linestyle=':', label='test-loss-x40', lw=2)
      plt.legend(loc=4, fontsize=legend_fontsize)

      if self.epoch_losses[:, 2].any():
        y_axis[:] = self.epoch_losses[:self.idx, 2]
        plt.plot(x_axis, y_axis*40, color='r', linestyle=':', label='valid-loss-x40', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

      if save_path is not None:
        fig.savefig('loss_'+save_path, dpi=dpi, bbox_inches='tight')
        print ('---- save figure {} into {}'.format(title, 'loss_'+save_path))
      plt.close(fig)
    
