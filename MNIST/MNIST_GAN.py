# Model based on:
# https://github.com/eriklindernoren/Keras-GAN/tree/master/gan

# -*- coding: utf-8 -*-
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Flatten
import numpy as np

channels = 1
latent_dim = 100
img_rows = 28
img_cols = 28
img_shape = (img_rows, img_cols, channels)


# def generator(input_dim=100,units=1024,activation='relu'):
def gan_generator():

    model = Sequential()

    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))

    model.summary()

    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img)


# def discriminator(input_shape=(28, 28, 1),nb_filter=64):
def gan_discriminator():

    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)
