# Based on:
# https://medium.com/@mrgarg.rajat/implementing-stackgan-using-keras-a0a1b381125e

import time
import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from . import CUB_StackGAN, CUB_dataset
from constants import DATA_DIRECTORY,OUTPUT_DIRECTORY,WEIGHTS_DIRECTORY,ROOT_DIR
from tqdm import tqdm

# Default parameters
BATCH_SIZE = 32
NUM_EPOCH = 100
STARTING_EPOCH = 1

def KL_loss(y_true, y_pred):
    mean = y_pred[:, :128]
    logsigma = y_pred[:, :128]
    loss = -logsigma + .5 * (-1 + K.exp(2. * logsigma) + K.square(mean))
    loss = K.mean(loss)
    return loss


def custom_generator_loss(y_true, y_pred):
    # Calculate binary cross entropy loss
    return K.binary_crossentropy(y_true, y_pred)


def save_rgb_img(img, path):
    """
    Save an rgb image
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Image")

    plt.savefig(path)
    plt.close()


def write_log(callback, name, loss, batch_no):
    """
    Write training summary to TensorBoard
    """
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = loss
    summary_value.tag = name
    callback.writer.add_summary(summary, batch_no)

    callback.writer.flush()


def save_weights(model, generator, discriminator, stage=1, epoch=0):
    generator.save_weights(
        os.path.join(ROOT_DIR + '/CUB' + WEIGHTS_DIRECTORY, model + '_CUB_stage' + str(stage)
                     + '_generator_weights_epoch_' + str(epoch) + '.h5'))
    discriminator.save_weights(
        os.path.join(ROOT_DIR + '/CUB' + WEIGHTS_DIRECTORY, model + '_CUB_stage' + str(stage)
                     + '_discriminator_weights_epoch_' + str(epoch) + '.h5'))


def load_weights(model, generator, discriminator, stage=1, epoch=0):
    try:
        generator.load_weights(
            os.path.join(ROOT_DIR + '/CUB' + WEIGHTS_DIRECTORY, model + '_CUB_stage' + str(stage)
                         + '_generator_weights_epoch_' + str(epoch) + '.h5'))
        discriminator.load_weights(
            os.path.join(ROOT_DIR + '/CUB' + WEIGHTS_DIRECTORY, model + '_CUB_stage' + str(stage)
                         + '_discriminator_weights_epoch_' + str(epoch) + '.h5'))
    except:
        print('Pre-trained weights not found, creating new files')
        save_weights(model, generator, discriminator)


def train(starting_epoch=STARTING_EPOCH, epochs=NUM_EPOCH, batch_size=BATCH_SIZE):

    data_dir = ROOT_DIR + '/CUB' + DATA_DIRECTORY
    train_dir = data_dir + "/train"
    test_dir = data_dir + "/test"
    image_size = 64
    z_dim = 100
    stage1_generator_lr = 0.0002
    stage1_discriminator_lr = 0.0002
    stage1_lr_decay_step = 600
    condition_dim = 128

    embeddings_file_path_train = train_dir + "/char-CNN-RNN-embeddings.pickle"
    embeddings_file_path_test = test_dir + "/char-CNN-RNN-embeddings.pickle"

    filenames_file_path_train = train_dir + "/filenames.pickle"
    filenames_file_path_test = test_dir + "/filenames.pickle"

    class_info_file_path_train = train_dir + "/class_info.pickle"
    class_info_file_path_test = test_dir + "/class_info.pickle"

    cub_dataset_dir = data_dir + "/CUB_200_2011"

    # Define optimizers
    dis_optimizer = Adam(lr=stage1_discriminator_lr, beta_1=0.5, beta_2=0.999)
    gen_optimizer = Adam(lr=stage1_generator_lr, beta_1=0.5, beta_2=0.999)

    """"
    Load datasets
    """
    print('Loading CUB Dataset...')
    X_train, y_train, embeddings_train = CUB_dataset.load_dataset(filenames_file_path=filenames_file_path_train,
                                                      class_info_file_path=class_info_file_path_train,
                                                      cub_dataset_dir=cub_dataset_dir,
                                                      embeddings_file_path=embeddings_file_path_train,
                                                      image_size=(64, 64))

    X_test, y_test, embeddings_test =  CUB_dataset.load_dataset(filenames_file_path=filenames_file_path_test,
                                                   class_info_file_path=class_info_file_path_test,
                                                   cub_dataset_dir=cub_dataset_dir,
                                                   embeddings_file_path=embeddings_file_path_test,
                                                   image_size=(64, 64))

    """
    Build and compile networks
    """

    print('Building StackGAN Network...')

    ca_model = CUB_StackGAN.build_ca_model()
    ca_model.compile(loss="binary_crossentropy", optimizer="adam")

    stage1_dis = CUB_StackGAN.build_stage1_discriminator()
    stage1_dis.compile(loss='binary_crossentropy', optimizer=dis_optimizer)

    stage1_gen = CUB_StackGAN.build_stage1_generator()
    stage1_gen.compile(loss="mse", optimizer=gen_optimizer)

    embedding_compressor_model = CUB_StackGAN.build_embedding_compressor_model()
    embedding_compressor_model.compile(loss="binary_crossentropy", optimizer="adam")

    adversarial_model = CUB_StackGAN.build_adversarial_model(gen_model=stage1_gen, dis_model=stage1_dis)
    adversarial_model.compile(loss=['binary_crossentropy', KL_loss], loss_weights=[1, 2.0],
                              optimizer=gen_optimizer, metrics=None)

    tensorboard = TensorBoard(log_dir="./logs".format(time.time()))
    tensorboard.set_model(stage1_gen)
    tensorboard.set_model(stage1_dis)
    tensorboard.set_model(ca_model)
    tensorboard.set_model(embedding_compressor_model)


    # Loading pretrained weights
    print('Loading pretrained weights...')
    load_weights(model='STACKGAN', generator=stage1_gen, discriminator=stage1_dis, stage=1, epoch=starting_epoch)

    # Generate an array containing real and fake values
    # Apply label smoothing as well
    real_labels = np.ones((batch_size, 1), dtype=float) * 0.9
    fake_labels = np.zeros((batch_size, 1), dtype=float) * 0.1

    print('Training configuration completed. Starting training epochs')

    for epoch in range(starting_epoch, epochs):
        print("========================================")
        print("Epoch is:", epoch)
        print("Number of batches", int(X_train.shape[0] / batch_size))

        gen_losses = []
        dis_losses = []

        # Load data and train model
        number_of_batches = int(X_train.shape[0] / batch_size)
        batch_iteration = tqdm(range(number_of_batches))
        batch_iteration.set_description(
            "Processing Epoch " + str(epoch) + " [" + str(epoch) + "/" + str(NUM_EPOCH) + "]")
        for index in batch_iteration:
            print("\nBatch:{}".format(index + 1))

            """
            Train the discriminator network
            """
            # Sample a batch of data
            z_noise = np.random.normal(0, 1, size=(batch_size, z_dim))
            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            embedding_batch = embeddings_train[index * batch_size:(index + 1) * batch_size]
            image_batch = (image_batch - 127.5) / 127.5

            # Generate fake images
            fake_images, _ = stage1_gen.predict([embedding_batch, z_noise], verbose=3)

            # Generate compressed embeddings
            compressed_embedding = embedding_compressor_model.predict_on_batch(embedding_batch)
            compressed_embedding = np.reshape(compressed_embedding, (-1, 1, 1, condition_dim))
            compressed_embedding = np.tile(compressed_embedding, (1, 4, 4, 1))

            dis_loss_real = stage1_dis.train_on_batch([image_batch, compressed_embedding],
                                                      np.reshape(real_labels, (batch_size, 1)))
            dis_loss_fake = stage1_dis.train_on_batch([fake_images, compressed_embedding],
                                                      np.reshape(fake_labels, (batch_size, 1)))
            dis_loss_wrong = stage1_dis.train_on_batch([image_batch[:(batch_size - 1)], compressed_embedding[1:]],
                                                       np.reshape(fake_labels[1:], (batch_size - 1, 1)))

            d_loss = 0.5 * np.add(dis_loss_real, 0.5 * np.add(dis_loss_wrong, dis_loss_fake))

            print("d_loss_real:{}".format(dis_loss_real))
            print("d_loss_fake:{}".format(dis_loss_fake))
            print("d_loss_wrong:{}".format(dis_loss_wrong))
            print("d_loss:{}".format(d_loss))

            """
            Train the generator network 
            """
            g_loss = adversarial_model.train_on_batch([embedding_batch, z_noise, compressed_embedding],
                                                      [K.ones((batch_size, 1)) * 0.9, K.ones((batch_size, 256)) * 0.9])
            print("g_loss:{}".format(g_loss))

            dis_losses.append(d_loss)
            gen_losses.append(g_loss)

        """
        Save losses to Tensorboard after each epoch
        """
        write_log(tensorboard, 'discriminator_loss', np.mean(dis_losses), epoch)
        write_log(tensorboard, 'generator_loss', np.mean(gen_losses[0]), epoch)

        # Generate and save images after every 5 epochs
        if epoch % 5 == 0:
            # z_noise2 = np.random.uniform(-1, 1, size=(batch_size, z_dim))
            z_noise2 = np.random.normal(0, 1, size=(batch_size, z_dim))
            embedding_batch = embeddings_test[0:batch_size]
            fake_images, _ = stage1_gen.predict_on_batch([embedding_batch, z_noise2])

            # Save images
            for i, img in enumerate(fake_images[:10]):
                save_rgb_img(img, OUTPUT_DIRECTORY + "/gen_{}_{}.png".format(epoch, i))
            save_weights(model='STACKGAN',generator=stage1_gen,discriminator=stage1_dis,stage=1,epoch=epoch)

    # Save models
    stage1_gen.save_weights("stage1_gen.h5")
    stage1_dis.save_weights("stage1_dis.h5")


if __name__ == '__main__':
    train()
