# Model based on:
# https://github.com/4thgen/DCGAN-CIFAR10/blob/master/GAN.py

# -*- coding: utf-8 -*-
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Reshape, Input, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Flatten, Dropout

import tensorflow as tf
import keras

channels = 3
latent_dim = 100
img_rows = 32
img_cols = 32
img_shape = (img_rows, img_cols, channels)


# def generator(input_dim=100,units=1024,activation='relu'):
def sagan_generator():
    model = Sequential()
    model.add(Dense(128 * 8 * 8, activation="relu", input_dim=latent_dim))
    model.add(Reshape((8, 8, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(channels, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))
    model.summary()
    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img)


# def discriminator(input_shape=(28, 28, 1),nb_filter=64):
def sagan_discriminator():

    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    # model.add(Self_Attention(1024))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)


# Based on:
# https://github.com/lysecret2/Self-Attention-Keras/blob/master/Attention_Example.ipynb


class Attention(keras.layers.Layer):

    def __init__(self, key_dim=None, **kwargs):
        self.key_dim = key_dim
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Weights initializer function
        w_initializer = keras.initializers.glorot_uniform()

        # Biases initializer function
        b_initializer = keras.initializers.Zeros()

        # Matrix to extract the keys
        self.key_extract = self.add_weight(name='feature_extract',
                                           shape=(int(input_shape[2]), int(self.key_dim)),
                                           initializer=w_initializer,
                                           trainable=True)
        # Key Bias
        self.key_bias = self.add_weight(name='feaure_bias',
                                        shape=(int(1), int(self.key_dim)),
                                        initializer=b_initializer,
                                        trainable=True)

        # The Query representing the class
        self.Query = self.add_weight(name='Query',
                                     shape=(int(self.key_dim), int(1)),
                                     initializer=w_initializer,
                                     trainable=True)

        super(Attention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # Extract the Keys
        keys = tf.tensordot(inputs, self.key_extract, axes=[2, 0]) + self.key_bias

        # Calculate the similarity between keys and the Query
        similar_logits = tf.tensordot(keys, self.Query, axes=[2, 0])

        # Normalize it to be between 0 and 1 and sum to 1
        attention_weights = tf.nn.softmax(similar_logits, axis=1)

        # Use these Weights to aggregate
        weighted_input = tf.matmul(inputs, attention_weights, transpose_a=True)

        return weighted_input

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2], int(1))


class Self_Attention(keras.layers.Layer):

    def __init__(self, key_dim=None, **kwargs):
        self.key_dim = key_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Weights initializer function
        w_initializer = keras.initializers.glorot_uniform()

        # Biases initializer function
        b_initializer = keras.initializers.Zeros()

        # Matrix to extract the keys
        self.key_extract = self.add_weight(name='feature_extract',
                                           shape=(int(input_shape[2]), int(self.key_dim)),
                                           initializer=w_initializer,
                                           trainable=True)
        # Key Bias
        self.key_bias = self.add_weight(name='feaure_bias',
                                        shape=(int(1), int(self.key_dim)),
                                        initializer=b_initializer,
                                        trainable=True)

        # The Query representing the class
        self.query_extract = self.add_weight(name='q_extract',
                                             shape=(int(input_shape[2]), int(self.key_dim)),
                                             initializer=w_initializer,
                                             trainable=True)
        self.query_bias = self.add_weight(name='q_bias',
                                          shape=(int(1), int(self.key_dim)),
                                          initializer=b_initializer,
                                          trainable=True)

        super(Self_Attention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # Extract the Keys
        keys = tf.tensordot(inputs, self.key_extract, axes=[2, 0]) + self.key_bias
        # Extract the Keys
        query = tf.tensordot(inputs, self.query_extract, axes=[2, 0]) + self.query_bias

        # Calculate the similarity between keys and the Query
        similar_logits = tf.matmul(query, keys, transpose_b=True)

        # Normalize it to be between 0 and 1 and sum to 1
        attention_weights = tf.nn.softmax(similar_logits, axis=1)

        # Use these Weights to aggregate
        weighted_input = tf.matmul(attention_weights, inputs)

        return weighted_input

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],input_shape[2])
