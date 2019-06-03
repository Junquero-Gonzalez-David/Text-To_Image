# Based on:
# https://github.com/vwrs/dcgan-mnist/blob/master/train.py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras.datasets import cifar10
from PIL import Image
from CIFAR10.CIFAR10_DCGAN import dcgan_discriminator, dcgan_generator
from CIFAR10.CIFAR10_SAGAN import sagan_discriminator, sagan_generator
from keras.models import Sequential
from keras.optimizers import Adam
import math
from constants import *
from tqdm import tqdm
import numpy as np

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# Training Settings
BATCH_SIZE = 32
NUM_EPOCH = 100
LR = 0.0002  # initial learning rate
B1 = 0.5  # momentum term


def save_weights(model, generator, discriminator):
    generator.save_weights(
        os.path.join(ROOT_DIR + '/CIFAR10' + WEIGHTS_DIRECTORY, model + '_cifar10_generator_weights.h5'))
    discriminator.save_weights(
        os.path.join(ROOT_DIR + '/CIFAR10' + WEIGHTS_DIRECTORY, model + '_cifar10_discriminator_weights.h5'))


def load_weights(model, generator, discriminator):
    try:
        generator.load_weights(
            os.path.join(ROOT_DIR + '/CIFAR10' + WEIGHTS_DIRECTORY, model + '_cifar10_generator_weights.h5'))
        discriminator.load_weights(
            os.path.join(ROOT_DIR + '/CIFAR10' + WEIGHTS_DIRECTORY, model + '_cifar10_discriminator_weights.h5'))
    except:
        print('Pre-trained weights not found, creating new files')
        save_weights(model, generator, discriminator)


def train(model, epochs=NUM_EPOCH, batch_size=BATCH_SIZE):

    NUM_EPOCH = epochs
    BATCH_SIZE = batch_size

    # Low-level TensorFlow configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.Session(config=config)
    set_session(sess)

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.astype(np.float32) - 127.5) / 127.5

    # build GAN model
    if model is MODEL_DCGAN:
        g = dcgan_generator()
        d = dcgan_discriminator()
    elif model is MODEL_SAGAN:
        g = sagan_generator()
        d = sagan_discriminator()
    else:
        print("Model " + model + " not implemented for the CIFAR10 dataset")
        return

    # Loading Pre-Trained weights
    load_weights(model=model, generator=g, discriminator=d)

    opt = Adam(lr=LR,beta_1=B1)
    d.trainable = True
    d.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer=opt)
    d.trainable = False
    dcgan = Sequential([g, d])
    opt= Adam(lr=LR,beta_1=B1)
    dcgan.compile(loss='binary_crossentropy',
                  metrics=['accuracy'],
                  optimizer=opt)

    num_batches = int(X_train.shape[0] / BATCH_SIZE)
    # create directory
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.mkdir(OUTPUT_DIRECTORY)
    if not os.path.exists(WEIGHTS_DIRECTORY):
        os.mkdir(WEIGHTS_DIRECTORY)

    print("-------------------")
    print("Total epoch:", NUM_EPOCH, "Number of batches:", num_batches)
    print("-------------------")
    z_pred = np.array([np.random.uniform(-1,1,100) for _ in range(49)])
    y_g = [1]*BATCH_SIZE
    y_d_true = [1]*BATCH_SIZE
    y_d_gen = [0]*BATCH_SIZE

    for epoch in list(map(lambda x: x + 1, range(NUM_EPOCH))):
        batch_iteration = tqdm(range(num_batches))
        batch_iteration.set_description("Processing Epoch " + str(epoch) +" [" + str(epoch) + "/" + str(NUM_EPOCH) + "]")
        for index in batch_iteration:
            X_d_true = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            X_g = np.array([np.random.normal(0,0.5,100) for _ in range(BATCH_SIZE)])
            X_d_gen = g.predict(X_g, verbose=0)

            # train discriminator
            d_loss = d.train_on_batch(X_d_true, y_d_true)
            d_loss = d.train_on_batch(X_d_gen, y_d_gen)
            # train generator
            g_loss = dcgan.train_on_batch(X_g, y_g)
            # show_progress(epoch,index,g_loss[0],d_loss[0],g_loss[1],d_loss[1])

        # save generated images
        image = combine_images(g.predict(z_pred))
        image = image*127.5 + 127.5
        Image.fromarray(image.astype(np.uint8)) \
            .save(os.path.join(ROOT_DIR + '/CIFAR10' + OUTPUT_DIRECTORY, model + '_cifar10_epoch_' + str(epoch) + '.png'))
        # save models
        save_weights(model=model, generator=g, discriminator=d)
        print("Epoch " + str(epoch) + " completed successfully")


def combine_images(generated_images):
    total, width, height, channels = generated_images.shape
    cols = int(math.sqrt(total))
    rows = math.ceil(float(total)/cols)
    combined_image = np.zeros((height*rows, width*cols, channels),
                              dtype=generated_images.dtype)

    for index, image in enumerate(generated_images):
        i = int(index/cols)
        j = index % cols
        combined_image[width*i:width*(i+1), height*j:height*(j+1)] = image[:, :, :]
    return combined_image

if __name__ == '__main__':
    train(model=MODEL_GAN)

