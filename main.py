from MNIST import MNIST_training
from CIFAR10 import CIFAR10_training
from constants import *

# Arguments
model = MODEL_DCGAN
dataset = CIFAR10

if __name__ == '__main__':
    if dataset is MNIST:
        MNIST_training.train(model)
    elif dataset is CIFAR10:
        CIFAR10_training.train(model)
