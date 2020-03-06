#!/usr/bin/env python3
# coding: utf-8

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import IPython
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K
from tensorflow.python.framework.ops import disable_eager_execution

# Import data from TIMIT dataset
data_dir = "data"
train_csv = pd.read_csv(os.path.join(data_dir, "train_data.csv"))
train_data = train_csv[train_csv.path_from_data_dir.str.contains('WAV.wav',  na=False)]
test_csv = pd.read_csv(os.path.join(data_dir, "test_data.csv"))
test_data = test_csv[test_csv.path_from_data_dir.str.contains('WAV.wav',  na=False)]
batch_size = 32
num_samples = 10
sample_rate = 16000

# Print statistics
print("Total number of training examples = " + str(train_data.shape[0]))
print("Total number of test examples = " + str(test_data.shape[0]))

raw_audio = tf.io.read_file(os.path.join(data_dir, train_data.path_from_data_dir[0]))
audio, sample_rate = tf.audio.decode_wav(raw_audio)
print("Shape of the audio file:", audio.shape)
print("Sample rate of waveform:", sample_rate)

# We can obtain the length of the audio file in seconds by doing audio.shape / sample_rate

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
        stfts = tf.signal.stft(signals, frame_length=1024, frame_step=256, fft_length=1024)
        spectrograms = tf.abs(stfts)

        # warp the linear scale spectrograms into the mel-scale
        num_spectrogram_bins = stfts.shape[-1]
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20.0, 8000.0, 128
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

        numpy_specgrams.append(mel_spectrograms)

    return np.array(pad(numpy_specgrams))

x_train = load_dataset_mel_spectogram(dataset=train_data)
x_test = load_dataset_mel_spectogram(dataset=test_data)

# We split training set into two halfs.
secret_audio_files = x_train[0:x_train.shape[0] // 2]
cover_audio_files = x_train[x_train.shape[0] // 2:]