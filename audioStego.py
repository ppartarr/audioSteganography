#!/usr/bin/env python3
# coding: utf-8

import datetime
import numpy as np
import os
import pandas as pd
import sys

# tensorflow
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, concatenate, GaussianNoise
from tensorflow.keras.models import Model
from tensorflow.keras import losses
from tensorflow.keras.optimizers import SGD

# memory investigation
from pympler import muppy, summary

# logging
import logging as log
log.basicConfig(format='%(asctime)s.%(msecs)06d: %(message)s',
                datefmt='%Y-%m-%d %I:%M:%S', level=log.INFO)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# regulate tensorflow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# import TIMIT dataset
data_dir = "data"
train_csv = pd.read_csv(os.path.join(data_dir, "train_data.csv"))
train_data = train_csv[train_csv.path_from_data_dir.str.contains(
    'WAV.wav', na=False)]
test_csv = pd.read_csv(os.path.join(data_dir, "test_data.csv"))
test_data = test_csv[test_csv.path_from_data_dir.str.contains(
    'WAV.wav', na=False)]
epochs = 3
batch_size = 32
num_samples = 1000
sample_rate = 16000
num_mel_filters = 16

# configuration statistics
log.info('Training examples: {}'.format(train_data.shape[0]))
log.info('Test examples: {}'.format(test_data.shape[0]))

# single sample informations
# the length of the audio file in seconds is audio.shape / sample_rate
audio, sample_rate = tf.audio.decode_wav(tf.io.read_file(os.path.join(
    data_dir, train_data.path_from_data_dir[0])))
log.info('Shape of the audio file: {}'.format(audio.shape))
log.info('Sample rate of waveform: {}'.format(sample_rate))
del audio
del sample_rate

# tensorboard visualisation
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)


def pad(dataset=train_data, padding_mode='CONSTANT'):

    padded_specgrams = []

    # padding
    pad_to = max([specgram.shape[1] for specgram in dataset])
    for specgram in dataset:
        pad_by = pad_to - specgram.shape[1]
        paddings = tf.constant([[0, 0], [0, pad_by], [0, 0]])
        specgram_pad = tf.pad(specgram, paddings, padding_mode)
        padded_specgrams.append(specgram_pad.numpy())

    return padded_specgrams


def load_dataset_mel_spectogram(num_audio_files=100, dataset=train_data):
    """
    Loads training and test datasets, from TIMIT and convert into spectrogram using STFT
    Arguments:
        num_audio_samples_per_class_train: number of audio per class to load into training dataset
    """

    # list initialization
    numpy_specgrams = []

    # data parsing
    while len(numpy_specgrams) < num_audio_files:
        sample = dataset.path_from_data_dir.sample()

        # extract audio and sample rate from WAV file
        raw_audio = tf.io.read_file(os.path.join(data_dir, sample.item()))
        audio, sample_rate = tf.audio.decode_wav(raw_audio)

        # reshape the signal to the shape of (batch_size, samples])
        signals = tf.reshape(audio, [1, -1])

        # a 1024-point STFT with frames of 64 ms and 75% overlap
        stfts = tf.signal.stft(signals, frame_length=1024,
                               frame_step=256, fft_length=1024)
        spectrograms = tf.abs(stfts)

        # warp the linear scale spectrograms into the mel-scale
        num_spectrogram_bins = stfts.shape[-1]
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20.0, 8000.0, num_mel_filters
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
        mel_spectrograms = tf.tensordot(
            spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(
            spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        numpy_specgrams.append(mel_spectrograms)

    return np.array(pad(numpy_specgrams))


# shape of the data is (n, 1, x, 128) where
#   n is the number of audio files
#   1 is the number of channels (mono)
#   x is the number of 64ms spectrograms with 716% overlap
#   128 is the number of mel filters
x_train = load_dataset_mel_spectogram(
    num_audio_files=num_samples, dataset=train_data)
x_test = load_dataset_mel_spectogram(
    num_audio_files=num_samples, dataset=test_data)
del train_csv
del train_data
del test_csv
del test_data

# we split training set into two halfs.
secret_audio_files = x_train[0:x_train.shape[0] // 2]
cover_audio_files = x_train[x_train.shape[0] // 2:]

summary.print_(summary.summarize(muppy.get_objects()))


def make_encoder(input_size):
    """ Returns the encoder as a Keras model, composed by Preparation and Hiding Networks """
    input_S = Input(shape=(input_size))
    input_C = Input(shape=(input_size))
    # print(input_S.shape)
    # print(input_C.shape)

    # Preparation Network
    x3 = Conv2D(num_mel_filters // 2, (num_mel_filters), padding='same', activation='sigmoid',
                name='conv_prep0_3x3')(input_S)
    x4 = Conv2D(num_mel_filters // 4, (num_mel_filters // 2), padding='same', activation='sigmoid',
                name='conv_prep0_4x4')(input_S)
    x5 = Conv2D(num_mel_filters // 8, (num_mel_filters // 4), padding='same', activation='sigmoid',
                name='conv_prep0_5x5')(input_S)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(num_mel_filters // 2, (num_mel_filters), padding='same',
                activation='sigmoid', name='conv_prep1_3x3')(x)
    x4 = Conv2D(num_mel_filters // 4, (num_mel_filters // 2), padding='same',
                activation='sigmoid', name='conv_prep1_4x4')(x)
    x5 = Conv2D(num_mel_filters // 8, (num_mel_filters // 4), padding='same',
                activation='sigmoid', name='conv_prep1_5x5')(x)
    x = concatenate([x3, x4, x5])

    x = concatenate([input_C, x])

    # Hiding network
    x3 = Conv2D(num_mel_filters // 2, (2, num_mel_filters), padding='same', activation='sigmoid',
                name='conv_hid0_3x3')(x)
    x4 = Conv2D(num_mel_filters // 4, (num_mel_filters // 2), padding='same', activation='sigmoid',
                name='conv_hid0_4x4')(x)
    x5 = Conv2D(num_mel_filters // 8, (num_mel_filters // 4), padding='same', activation='sigmoid',
                name='conv_hid0_5x5')(x)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(num_mel_filters // 2, (2, num_mel_filters), padding='same',
                activation='sigmoid', name='conv_hid1_3x3')(x)
    x4 = Conv2D(num_mel_filters // 4, (num_mel_filters // 2), padding='same', activation='sigmoid',
                name='conv_hid1_4x4')(x)
    x5 = Conv2D(num_mel_filters // 8, (num_mel_filters // 4), padding='same', activation='sigmoid',
                name='conv_hid1_5x5')(x)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(num_mel_filters // 2, (2, num_mel_filters), padding='same',
                activation='sigmoid', name='conv_hid2_3x3')(x)
    x4 = Conv2D(num_mel_filters // 4, (2, num_mel_filters), padding='same',
                activation='sigmoid', name='conv_hid2_4x4')(x)
    x5 = Conv2D(num_mel_filters // 8, (2, num_mel_filters), padding='same',
                activation='sigmoid', name='conv_hid2_5x5')(x)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(num_mel_filters // 2, (2, num_mel_filters), padding='same',
                activation='sigmoid', name='conv_hid3_3x3')(x)
    x4 = Conv2D(num_mel_filters // 4, (num_mel_filters // 2), padding='same', activation='sigmoid',
                name='conv_hid3_4x4')(x)
    x5 = Conv2D(num_mel_filters // 8, (num_mel_filters // 4), padding='same', activation='sigmoid',
                name='conv_hid3_5x5')(x)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(num_mel_filters // 2, (2, num_mel_filters), padding='same',
                activation='sigmoid', name='conv_hid4_3x3')(x)
    x4 = Conv2D(num_mel_filters // 4, (num_mel_filters // 2), padding='same', activation='sigmoid',
                name='conv_hid4_4x4')(x)
    x5 = Conv2D(num_mel_filters // 8, (num_mel_filters // 4), padding='same', activation='sigmoid',
                name='conv_hid4_5x5')(x)
    x = concatenate([x3, x4, x5])

    output_Cprime = Conv2D(num_mel_filters, (2, num_mel_filters), padding='same',
                           activation='sigmoid', name='output_C')(x)

    return Model(inputs=[input_S, input_C],
                 outputs=output_Cprime,
                 name='Encoder')


def make_decoder(input_size):
    """ Returns the decoder as a Keras model, composed by the Reveal Network """
    # Reveal network
    reveal_input = Input(shape=(input_size))
    # print(reveal_input.shape)

    # Adding Gaussian noise with 0.01 standard deviation.
    input_with_noise = GaussianNoise(0.01, name='output_C_noise')(reveal_input)
    # print(input_with_noise.shape)

    x3 = Conv2D(num_mel_filters // 2, (2, num_mel_filters), padding='same', activation='sigmoid',
                name='conv_rev0_3x3')(input_with_noise)
    x4 = Conv2D(num_mel_filters // 4, (num_mel_filters // 2), padding='same', activation='sigmoid',
                name='conv_rev0_4x4')(input_with_noise)
    x5 = Conv2D(num_mel_filters // 8, (num_mel_filters // 4), padding='same', activation='sigmoid',
                name='conv_rev0_5x5')(input_with_noise)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(num_mel_filters // 2, (2, num_mel_filters), padding='same',
                activation='sigmoid', name='conv_rev1_3x3')(x)
    x4 = Conv2D(num_mel_filters // 4, (num_mel_filters // 2), padding='same', activation='sigmoid',
                name='conv_rev1_4x4')(x)
    x5 = Conv2D(num_mel_filters // 8, (num_mel_filters // 4), padding='same', activation='sigmoid',
                name='conv_rev1_5x5')(x)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(num_mel_filters // 2, (2, num_mel_filters), padding='same',
                activation='sigmoid', name='conv_rev2_3x3')(x)
    x4 = Conv2D(num_mel_filters // 4, (num_mel_filters // 2), padding='same', activation='sigmoid',
                name='conv_rev2_4x4')(x)
    x5 = Conv2D(num_mel_filters // 8, (num_mel_filters // 4), padding='same', activation='sigmoid',
                name='conv_rev2_5x5')(x)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(num_mel_filters // 2, (2, num_mel_filters), padding='same',
                activation='sigmoid', name='conv_rev3_3x3')(x)
    x4 = Conv2D(num_mel_filters // 4, (num_mel_filters // 2), padding='same', activation='sigmoid',
                name='conv_rev3_4x4')(x)
    x5 = Conv2D(num_mel_filters // 8, (num_mel_filters // 4), padding='same', activation='sigmoid',
                name='conv_rev3_5x5')(x)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(num_mel_filters // 2, (2, num_mel_filters), padding='same',
                activation='sigmoid', name='conv_rev4_3x3')(x)
    x4 = Conv2D(num_mel_filters // 4, (num_mel_filters // 2), padding='same', activation='sigmoid',
                name='conv_rev4_4x4')(x)
    x5 = Conv2D(num_mel_filters // 8, (num_mel_filters // 4), padding='same', activation='sigmoid',
                name='conv_rev4_5x5')(x)
    x = concatenate([x3, x4, x5])

    output_Sprime = Conv2D(num_mel_filters, (2, num_mel_filters), padding='same',
                           activation='sigmoid', name='output_S')(x)

    return Model(inputs=reveal_input,
                 outputs=output_Sprime,
                 name='Decoder')


def make_model(input_size):
    """ Full model """
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    input_S = Input(shape=(input_size))
    input_C = Input(shape=(input_size))
    # print(input_S.shape)
    # print(input_C.shape)

    encoder = make_encoder(input_size)

    decoder = make_decoder(input_size)
    decoder.compile(optimizer=sgd,
                    loss=losses.binary_crossentropy, metrics=['accuracy'])
    decoder.trainable = False

    output_Cprime = encoder([input_S, input_C])
    # print(output_Cprime.shape)
    output_Sprime = decoder(output_Cprime)

    autoencoder = Model(inputs=[input_S, input_C],
                        outputs=concatenate([output_Sprime, output_Cprime]))
    autoencoder.compile(optimizer=sgd,
                        loss=losses.binary_crossentropy, metrics=['accuracy'])

    return encoder, decoder, autoencoder


# print(type(secret_audio_files[0]))
# print(secret_audio_files.shape[1:])
encoder_model, reveal_model, autoencoder_model = make_model(
    secret_audio_files.shape[1:])


def lr_schedule(epoch_idx):
    if epoch_idx < 200:
        return 0.001
    elif epoch_idx < 400:
        return 0.0003
    elif epoch_idx < 600:
        return 0.0001
    else:
        return 0.00003


x_data = [secret_audio_files, cover_audio_files]
y_data = np.concatenate((secret_audio_files, cover_audio_files), axis=3)

if len(sys.argv) == 1:
    autoencoder_model.fit(x=x_data, y=y_data, epochs=epochs,
                          batch_size=batch_size, callbacks=[tensorboard_callback])
    # num_secret_audio_files = secret_audio_files.shape[0]
    # loss_history = []
    # for epoch in range(epochs):
    #     np.random.shuffle(secret_audio_files)
    #     np.random.shuffle(cover_audio_files)

    #     t = tqdm(range(0, num_secret_audio_files, batch_size), mininterval=0)
    #     ae_loss = []
    #     rev_loss = []
    #     for idx in t:

    #         batch_S = secret_audio_files[idx:min(idx + batch_size, num_secret_audio_files)]
    #         batch_C = cover_audio_files[idx:min(idx + batch_size, num_secret_audio_files)]

    #         C_prime = encoder_model.predict([batch_S, batch_C])

    #         ae_loss.append(autoencoder_model.train_on_batch(x=[batch_S, batch_C],
    #                                                         y=np.concatenate((batch_S, batch_C), axis=3)))
    #         rev_loss.append(reveal_model.train_on_batch(x=C_prime,
    #                                                     y=batch_S))

    #         # Update learning rate
    #         K.set_value(autoencoder_model.optimizer.lr, lr_schedule(epoch))
    #         K.set_value(reveal_model.optimizer.lr, lr_schedule(epoch))
    #         t.set_description('Epoch {} | Batch: {} of {}. Loss AE {:10.2f} | Loss Rev {:10.2f}'.format(
    #             epoch + 1, idx, num_secret_audio_files, np.mean(ae_loss), np.mean(rev_loss)))
    #     loss_history.append(np.mean(ae_loss))

    # save model
    model_hdf5 = 'model-{}.hdf5'.format(
        datetime.datetime.now().strftime("%Y%m%d_%H%M"))
    autoencoder_model.save_weights(model_hdf5)
    log.info('Model weights saved at {}'.format(model_hdf5))
else:
    autoencoder_model.load_weights(sys.argv[1])
    log.info('Model loaded from {}'.format(sys.argv[1]))
