import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_charts(modelLoc):
  fileObj = open("./" + modelLoc + "/log.txt", 'r')

  epoch = []
  trainLoss = []
  trainScore = []
  valLoss = []
  valScore = []

  # Read in File
  for line in fileObj:
    words = line.split()
    if words[0] == 'epoch':
      epoch.append(int(words[1][:-1]))
    elif words[0] == 'train_loss:':
      trainLoss.append(float(words[1][:-1]))
      trainScore.append(float(words[3]))
    elif words[0] == 'eval':
      valLoss.append(float(words[2][:-1]))
      valScore.append(float(words[4]))

  fileObj.close()

  minValLoss = min(valLoss)
  epochValLoss = np.argmin(valLoss)
  maxValScore = max(valScore)
  epochValScore = np.argmax(valScore)

  # Plot Loss and Score
  plt.figure()
  plt.suptitle(modelLoc)

  # train/val loss
  plt.subplot(211)
  plt.plot(epoch, trainLoss, label='Training')
  plt.plot(epoch, valLoss, label='Validation')
  plt.plot(epochValLoss, minValLoss, marker='x', markersize=3, color="black")
  plt.xlabel('loss')
  plt.ylabel('score')
  plt.title('Training and Validation Loss')
  plt.legend()
  plt.minorticks_on()
  plt.grid(True, which='both')

  # train/val score
  plt.subplot(212)
  plt.plot(epoch, trainScore, label='Training')
  plt.plot(epoch, valScore, label='Validation')
  plt.plot(epochValScore, maxValScore, marker='x', markersize=3, color="black")
  plt.xlabel('epochs')
  plt.ylabel('score')
  plt.title('Training and Validation Score')
  plt.legend()
  plt.minorticks_on()
  plt.grid(True, which='both')

  plt.subplots_adjust(hspace=0.5)
  #plt.show()
  plt.savefig("./" + modelLoc + ".png")


if __name__ == '__main__':
  modelLoc = sys.argv[1]
  plot_charts(modelLoc)
