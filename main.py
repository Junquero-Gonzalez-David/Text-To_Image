from MNIST import MNIST_training
from CIFAR10 import CIFAR10_training
from CUB import CUB_training
from constants import *

# Arguments
model = MODEL_DCGAN
dataset = CUB

# ---------------------------

epochs = 1000
batch_size = 64

# Only for CUB
starting_epoch = 1

# ---------------------------

if __name__ == '__main__':
    if dataset is MNIST:
        MNIST_training.train(model = model,
                             epochs = epochs,
                             batch_size = batch_size
                             )
    elif dataset is CIFAR10:
        CIFAR10_training.train(model = model,
                             epochs = epochs,
                             batch_size = batch_size
                             )
    elif dataset is CUB:
        CUB_training.train(starting_epoch = starting_epoch,
                           epochs = epochs,
                           batch_size = batch_size
                           )
