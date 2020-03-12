#!/usr/bin/env python3
# coding: utf-8

import datetime
import numpy as np
import os
import pandas as pd
import sys

# tensorflow
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import Input, Conv2D, concatenate, GaussianNoise
from tensorflow.keras.models import Model
from tensorflow.keras import losses

# memory investigation
from pympler import muppy, summary

# logging
import logging as log
log.basicConfig(format='%(asctime)s.%(msecs)06d: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S', level=log.INFO)
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
epochs = 100
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


def steg_model(input_shape, pretrain=False):

    lossFns = {
        "hide_conv_f": losses.mean_squared_error,
        "revl_conv_f": losses.mean_squared_error,
    }
    lossWeights = {
        "hide_conv_f": 1.0,
        "revl_conv_f": 0.75
    }

    # Inputs
    secret = Input(shape=input_shape, name='secret')
    cover = Input(shape=input_shape, name='cover')

    # Prepare network - patches [3*3,4*4,5*5]
    pconv_3x3 = Conv2D(50, kernel_size=3, padding="same",
                       activation='relu', name='prep_conv3x3_1')(secret)
    pconv_3x3 = Conv2D(50, kernel_size=3, padding="same",
                       activation='relu', name='prep_conv3x3_2')(pconv_3x3)
    pconv_3x3 = Conv2D(50, kernel_size=3, padding="same",
                       activation='relu', name='prep_conv3x3_3')(pconv_3x3)
    pconv_3x3 = Conv2D(50, kernel_size=3, padding="same",
                       activation='relu', name='prep_conv3x3_4')(pconv_3x3)

    pconv_4x4 = Conv2D(50, kernel_size=4, padding="same",
                       activation='relu', name='prep_conv4x4_1')(secret)
    pconv_4x4 = Conv2D(50, kernel_size=4, padding="same",
                       activation='relu', name='prep_conv4x4_2')(pconv_4x4)
    pconv_4x4 = Conv2D(50, kernel_size=4, padding="same",
                       activation='relu', name='prep_conv4x4_3')(pconv_4x4)
    pconv_4x4 = Conv2D(50, kernel_size=4, padding="same",
                       activation='relu', name='prep_conv4x4_4')(pconv_4x4)

    pconv_5x5 = Conv2D(50, kernel_size=5, padding="same",
                       activation='relu', name='prep_conv5x5_1')(secret)
    pconv_5x5 = Conv2D(50, kernel_size=5, padding="same",
                       activation='relu', name='prep_conv5x5_2')(pconv_5x5)
    pconv_5x5 = Conv2D(50, kernel_size=5, padding="same",
                       activation='relu', name='prep_conv5x5_3')(pconv_5x5)
    pconv_5x5 = Conv2D(50, kernel_size=5, padding="same",
                       activation='relu', name='prep_conv5x5_4')(pconv_5x5)

    pconcat_1 = concatenate(
        [pconv_3x3, pconv_4x4, pconv_5x5], axis=3, name="prep_concat_1")

    pconv_5x5 = Conv2D(50, kernel_size=5, padding="same",
                       activation='relu', name='prep_conv5x5_f')(pconcat_1)
    pconv_4x4 = Conv2D(50, kernel_size=4, padding="same",
                       activation='relu', name='prep_conv4x4_f')(pconcat_1)
    pconv_3x3 = Conv2D(50, kernel_size=3, padding="same",
                       activation='relu', name='prep_conv3x3_f')(pconcat_1)

    pconcat_f1 = concatenate(
        [pconv_5x5, pconv_4x4, pconv_3x3], axis=3, name="prep_concat_2")

    # Hiding network - patches [3*3,4*4,5*5]
    hconcat_h = concatenate([cover, pconcat_f1], axis=3, name="hide_concat_1")

    hconv_3x3 = Conv2D(50, kernel_size=3, padding="same",
                       activation='relu', name='hide_conv3x3_1')(hconcat_h)
    hconv_3x3 = Conv2D(50, kernel_size=3, padding="same",
                       activation='relu', name='hide_conv3x3_2')(hconv_3x3)
    hconv_3x3 = Conv2D(50, kernel_size=3, padding="same",
                       activation='relu', name='hide_conv3x3_3')(hconv_3x3)
    hconv_3x3 = Conv2D(50, kernel_size=3, padding="same",
                       activation='relu', name='hide_conv3x3_4')(hconv_3x3)

    hconv_4x4 = Conv2D(50, kernel_size=4, padding="same",
                       activation='relu', name='hide_conv4x4_1')(hconcat_h)
    hconv_4x4 = Conv2D(50, kernel_size=4, padding="same",
                       activation='relu', name='hide_conv4x4_2')(hconv_4x4)
    hconv_4x4 = Conv2D(50, kernel_size=4, padding="same",
                       activation='relu', name='hide_conv4x4_3')(hconv_4x4)
    hconv_4x4 = Conv2D(50, kernel_size=4, padding="same",
                       activation='relu', name='hide_conv4x4_4')(hconv_4x4)

    hconv_5x5 = Conv2D(50, kernel_size=5, padding="same",
                       activation='relu', name='hide_conv5x5_1')(hconcat_h)
    hconv_5x5 = Conv2D(50, kernel_size=5, padding="same",
                       activation='relu', name='hide_conv5x5_2')(hconv_5x5)
    hconv_5x5 = Conv2D(50, kernel_size=5, padding="same",
                       activation='relu', name='hide_conv5x5_3')(hconv_5x5)
    hconv_5x5 = Conv2D(50, kernel_size=5, padding="same",
                       activation='relu', name='hide_conv5x5_4')(hconv_5x5)

    hconcat_1 = concatenate(
        [hconv_3x3, hconv_4x4, hconv_5x5], axis=3, name="hide_concat_2")

    hconv_5x5 = Conv2D(50, kernel_size=5, padding="same",
                       activation='relu', name='hide_conv5x5_f')(hconcat_1)
    hconv_4x4 = Conv2D(50, kernel_size=4, padding="same",
                       activation='relu', name='hide_conv4x4_f')(hconcat_1)
    hconv_3x3 = Conv2D(50, kernel_size=3, padding="same",
                       activation='relu', name='hide_conv3x3_f')(hconcat_1)

    hconcat_f1 = concatenate(
        [hconv_5x5, hconv_4x4, hconv_3x3], axis=3, name="hide_concat_3")

    cover_pred = Conv2D(num_mel_filters, kernel_size=1, padding="same",
                        name='hide_conv_f')(hconcat_f1)

    # Noise layer
    noise_ip = GaussianNoise(0.1)(cover_pred)

    # Reveal network - patches [3*3,4*4,5*5]
    rconv_3x3 = Conv2D(50, kernel_size=3, padding="same",
                       activation='relu', name='revl_conv3x3_1')(noise_ip)
    rconv_3x3 = Conv2D(50, kernel_size=3, padding="same",
                       activation='relu', name='revl_conv3x3_2')(rconv_3x3)
    rconv_3x3 = Conv2D(50, kernel_size=3, padding="same",
                       activation='relu', name='revl_conv3x3_3')(rconv_3x3)
    rconv_3x3 = Conv2D(50, kernel_size=3, padding="same",
                       activation='relu', name='revl_conv3x3_4')(rconv_3x3)

    rconv_4x4 = Conv2D(50, kernel_size=4, padding="same",
                       activation='relu', name='revl_conv4x4_1')(noise_ip)
    rconv_4x4 = Conv2D(50, kernel_size=4, padding="same",
                       activation='relu', name='revl_conv4x4_2')(rconv_4x4)
    rconv_4x4 = Conv2D(50, kernel_size=4, padding="same",
                       activation='relu', name='revl_conv4x4_3')(rconv_4x4)
    rconv_4x4 = Conv2D(50, kernel_size=4, padding="same",
                       activation='relu', name='revl_conv4x4_4')(rconv_4x4)

    rconv_5x5 = Conv2D(50, kernel_size=5, padding="same",
                       activation='relu', name='revl_conv5x5_1')(noise_ip)
    rconv_5x5 = Conv2D(50, kernel_size=5, padding="same",
                       activation='relu', name='revl_conv5x5_2')(rconv_5x5)
    rconv_5x5 = Conv2D(50, kernel_size=5, padding="same",
                       activation='relu', name='revl_conv5x5_3')(rconv_5x5)
    rconv_5x5 = Conv2D(50, kernel_size=5, padding="same",
                       activation='relu', name='revl_conv5x5_4')(rconv_5x5)

    rconcat_1 = concatenate(
        [rconv_3x3, rconv_4x4, rconv_5x5], axis=3, name="revl_concat_1")

    rconv_5x5 = Conv2D(50, kernel_size=5, padding="same",
                       activation='relu', name='revl_conv5x5_f')(rconcat_1)
    rconv_4x4 = Conv2D(50, kernel_size=4, padding="same",
                       activation='relu', name='revl_conv4x4_f')(rconcat_1)
    rconv_3x3 = Conv2D(50, kernel_size=3, padding="same",
                       activation='relu', name='revl_conv3x3_f')(rconcat_1)

    rconcat_f1 = concatenate(
        [rconv_5x5, rconv_4x4, rconv_3x3], axis=3, name="revl_concat_2")

    secret_pred = Conv2D(num_mel_filters, kernel_size=1, padding="same",
                         name='revl_conv_f')(rconcat_f1)

    model = Model(inputs=[secret, cover], outputs=[cover_pred, secret_pred])

    # Compile model
    model.compile(optimizer='adam', loss=lossFns, loss_weights=lossWeights)

    return model


# print(type(secret_audio_files[0]))
# print(secret_audio_files.shape[1:])
model = steg_model(cover_audio_files.shape[1:], pretrain=False)


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
    # callbacks
    callback_tensorboard = TensorBoard(
        log_dir=log_dir, histogram_freq=1)
    callback_checkpoint = ModelCheckpoint(
        log_dir, monitor='loss', verbose=1, save_best_only=True, mode='max')

    # train model
    model.fit(x=x_data, y=x_data, epochs=epochs,
              batch_size=batch_size, callbacks=[callback_tensorboard, callback_checkpoint])

    # save model
    model_hdf5 = 'model-{}.hdf5'.format(
        datetime.datetime.now().strftime("%Y%m%d_%H%M"))
    model.save_weights(model_hdf5)
    log.info('Model weights saved at {}'.format(model_hdf5))
else:
    model.load_weights(sys.argv[1])
    log.info('Model loaded from {}'.format(sys.argv[1]))
