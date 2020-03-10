#!/usr/bin/env python3
# coding: utf-8

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

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

# regulate tensorflow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# Import data from TIMIT dataset
data_dir = "data"
train_csv = pd.read_csv(os.path.join(data_dir, "train_data.csv"))
train_data = train_csv[train_csv.path_from_data_dir.str.contains(
    'WAV.wav', na=False)]
test_csv = pd.read_csv(os.path.join(data_dir, "test_data.csv"))
test_data = test_csv[test_csv.path_from_data_dir.str.contains(
    'WAV.wav', na=False)]
epochs = 1
batch_size = 32
num_samples = 10
sample_rate = 16000

# Print statistics
print("Total number of training examples = " + str(train_data.shape[0]))
print("Total number of test examples = " + str(test_data.shape[0]))

raw_audio = tf.io.read_file(os.path.join(
    data_dir, train_data.path_from_data_dir[0]))
audio, sample_rate = tf.audio.decode_wav(raw_audio)
print("Shape of the audio file:", audio.shape)
print("Sample rate of waveform:", sample_rate)

# We can obtain the length of the audio file in seconds by doing
# audio.shape / sample_rate


def pad(dataset=train_data, padding_mode="CONSTANT"):

    padded_specgrams = []

    # padding
    pad_to = max([specgram.shape[1] for specgram in dataset])
    for specgram in dataset:
        pad_by = pad_to - specgram.shape[1]
        paddings = tf.constant([[0, 0], [0, pad_by], [0, 0]])
        specgram_pad = tf.pad(specgram, paddings, padding_mode)
        padded_specgrams.append(specgram_pad.numpy())

    return padded_specgrams

# Convert audio to spectrogram using Short Time Fourier Transform (STFT)


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
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20.0, 8000.0, 128
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

# we split training set into two halfs.
secret_audio_files = x_train[0:x_train.shape[0] // 2]
cover_audio_files = x_train[x_train.shape[0] // 2:]

# Variable used to weight the losses of the secret and cover images (See
# paper for more details)
beta = 1.0

# Loss for reveal network


def rev_loss_fn(s_true, s_pred):
    # Loss for reveal network is: beta * |S-S'|
    return beta * losses.mean_squared_error(s_true, s_pred)

# Loss for the full model, used for preparation and hidding networks


def full_loss(y_true, y_pred):
    # print('y_true', y_true)
    # print('y_pred', y_pred)
    # Loss for the full model is: |C-C'| + beta * |S-S'|
    s_true, c_true = y_true[..., 0:128], y_true[..., 128:256]
    s_pred, c_pred = y_pred[..., 0:128], y_pred[..., 128:256]

    # print("y_true: {}".format(y_true))
    # print("y_pred: {}".format(y_pred))

    s_loss = rev_loss_fn(s_true, s_pred)
    c_loss = losses.mean_squared_error(c_true, c_pred)
    return s_loss + c_loss

# Returns the encoder as a Keras model, composed by Preparation and Hiding
# Networks.


def make_encoder(input_size):
    input_S = Input(shape=(input_size))
    input_C = Input(shape=(input_size))
    # print(input_S.shape)
    # print(input_C.shape)

    # Preparation Network
    x3 = Conv2D(64, (128), padding='same', activation='relu',
                name='conv_prep0_3x3')(input_S)
    x4 = Conv2D(32, (64), padding='same', activation='relu',
                name='conv_prep0_4x4')(input_S)
    x5 = Conv2D(16, (32), padding='same', activation='relu',
                name='conv_prep0_5x5')(input_S)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(64, (128), padding='same',
                activation='relu', name='conv_prep1_3x3')(x)
    x4 = Conv2D(32, (64), padding='same',
                activation='relu', name='conv_prep1_4x4')(x)
    x5 = Conv2D(16, (32), padding='same',
                activation='relu', name='conv_prep1_5x5')(x)
    x = concatenate([x3, x4, x5])

    x = concatenate([input_C, x])

    # Hiding network
    x3 = Conv2D(64, (2, 128), padding='same', activation='relu',
                name='conv_hid0_3x3')(x)
    x4 = Conv2D(32, (64), padding='same', activation='relu',
                name='conv_hid0_4x4')(x)
    x5 = Conv2D(16, (32), padding='same', activation='relu',
                name='conv_hid0_5x5')(x)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(64, (2, 128), padding='same',
                activation='relu', name='conv_hid1_3x3')(x)
    x4 = Conv2D(32, (64), padding='same', activation='relu',
                name='conv_hid1_4x4')(x)
    x5 = Conv2D(16, (32), padding='same', activation='relu',
                name='conv_hid1_5x5')(x)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(64, (2, 128), padding='same',
                activation='relu', name='conv_hid2_3x3')(x)
    x4 = Conv2D(32, (2, 128), padding='same',
                activation='relu', name='conv_hid2_4x4')(x)
    x5 = Conv2D(16, (2, 128), padding='same',
                activation='relu', name='conv_hid2_5x5')(x)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(64, (2, 128), padding='same',
                activation='relu', name='conv_hid3_3x3')(x)
    x4 = Conv2D(32, (64), padding='same', activation='relu',
                name='conv_hid3_4x4')(x)
    x5 = Conv2D(16, (32), padding='same', activation='relu',
                name='conv_hid3_5x5')(x)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(64, (2, 128), padding='same',
                activation='relu', name='conv_hid4_3x3')(x)
    x4 = Conv2D(32, (64), padding='same', activation='relu',
                name='conv_hid4_4x4')(x)
    x5 = Conv2D(16, (32), padding='same', activation='relu',
                name='conv_hid4_5x5')(x)
    x = concatenate([x3, x4, x5])

    output_Cprime = Conv2D(128, (2, 128), padding='same',
                           activation='relu', name='output_C')(x)

    return Model(inputs=[input_S, input_C],
                 outputs=output_Cprime,
                 name='Encoder')

# Returns the decoder as a Keras model, composed by the Reveal Network


def make_decoder(input_size):

    # Reveal network
    reveal_input = Input(shape=(input_size))
    # print(reveal_input.shape)

    # Adding Gaussian noise with 0.01 standard deviation.
    input_with_noise = GaussianNoise(0.01, name='output_C_noise')(reveal_input)
    # print(input_with_noise.shape)

    x3 = Conv2D(64, (2, 128), padding='same', activation='relu',
                name='conv_rev0_3x3')(input_with_noise)
    x4 = Conv2D(32, (64), padding='same', activation='relu',
                name='conv_rev0_4x4')(input_with_noise)
    x5 = Conv2D(16, (32), padding='same', activation='relu',
                name='conv_rev0_5x5')(input_with_noise)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(64, (2, 128), padding='same',
                activation='relu', name='conv_rev1_3x3')(x)
    x4 = Conv2D(32, (64), padding='same', activation='relu',
                name='conv_rev1_4x4')(x)
    x5 = Conv2D(16, (32), padding='same', activation='relu',
                name='conv_rev1_5x5')(x)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(64, (2, 128), padding='same',
                activation='relu', name='conv_rev2_3x3')(x)
    x4 = Conv2D(32, (64), padding='same', activation='relu',
                name='conv_rev2_4x4')(x)
    x5 = Conv2D(16, (32), padding='same', activation='relu',
                name='conv_rev2_5x5')(x)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(64, (2, 128), padding='same',
                activation='relu', name='conv_rev3_3x3')(x)
    x4 = Conv2D(32, (64), padding='same', activation='relu',
                name='conv_rev3_4x4')(x)
    x5 = Conv2D(16, (32), padding='same', activation='relu',
                name='conv_rev3_5x5')(x)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(64, (2, 128), padding='same',
                activation='relu', name='conv_rev4_3x3')(x)
    x4 = Conv2D(32, (64), padding='same', activation='relu',
                name='conv_rev4_4x4')(x)
    x5 = Conv2D(16, (32), padding='same', activation='relu',
                name='conv_rev4_5x5')(x)
    x = concatenate([x3, x4, x5])

    output_Sprime = Conv2D(128, (2, 128), padding='same',
                           activation='relu', name='output_S')(x)

    return Model(inputs=reveal_input,
                 outputs=output_Sprime,
                 name='Decoder')

# Full model.


def make_model(input_size):
    input_S = Input(shape=(input_size))
    input_C = Input(shape=(input_size))
    # print(input_S.shape)
    # print(input_C.shape)

    encoder = make_encoder(input_size)

    decoder = make_decoder(input_size)
    decoder.compile(optimizer='adam', loss=rev_loss_fn)
    decoder.trainable = False

    output_Cprime = encoder([input_S, input_C])
    # print(output_Cprime.shape)
    output_Sprime = decoder(output_Cprime)

    autoencoder = Model(inputs=[input_S, input_C],
                        outputs=concatenate([output_Sprime, output_Cprime]))
    autoencoder.compile(optimizer='adam', loss=full_loss)

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


if len(sys.argv) == 1:
    autoencoder_model.fit(x=[secret_audio_files, cover_audio_files], y=np.concatenate(
        (secret_audio_files, cover_audio_files), axis=3), epochs=epochs, batch_size=batch_size)

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
    #                                                         y=np.concatenate((secret_audio_files, cover_audio_files), axis=3)))
    #         rev_loss.append(reveal_model.train_on_batch(x=C_prime,
    #                                                     y=batch_S))

    #         # Update learning rate
    #         K.set_value(autoencoder_model.optimizer.lr, lr_schedule(epoch))
    #         K.set_value(reveal_model.optimizer.lr, lr_schedule(epoch))
    #         t.set_description('Epoch {} | Batch: {} of {}. Loss AE {:10.2f} | Loss Rev {:10.2f}'.format(
    #             epoch + 1, idx, num_secret_audio_files, np.mean(ae_loss), np.mean(rev_loss)))
    #     loss_history.append(np.mean(ae_loss))

    # save model
    autoencoder_model.save_weights(
        'model-{}.hdf5'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M")))
else:
    autoencoder_model.load_weights(sys.argv[1])
    print('Model loaded from', sys.argv[1])
