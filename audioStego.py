#!/usr/bin/env python3
# coding: utf-8

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from torch.utils.data import Dataset
import IPython
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

# Import data from TIMIT dataset
DATA_DIR = "data"
TRAIN_DATA = pd.read_csv(os.path.join(DATA_DIR, "train_data.csv"))
TRAIN_DATA = TRAIN_DATA[TRAIN_DATA.path_from_data_dir.str.contains('WAV.wav',  na=False)]
TEST_DATA = pd.read_csv(os.path.join(DATA_DIR, "test_data.csv"))
TEST_DATA = TEST_DATA[TEST_DATA.path_from_data_dir.str.contains('WAV.wav',  na=False)]

# Print example waveform & stats
print("Number of training examples:", TRAIN_DATA.shape[0])
print("Number of test examples:", TEST_DATA.shape[0])
waveform, sample_rate = torchaudio.load_wav(os.path.join(DATA_DIR, TRAIN_DATA.path_from_data_dir[0]))
# sample rate is the number of times per second the value of the audio signal is saved
# shape is the total number of samples
print("Shape of waveform:", waveform.shape)
print("Sample rate of waveform:", sample_rate)
print("Sample length:", int(waveform.size()[1]) / sample_rate)

# Convert audio to spectrogram using Short Time Fourier Transform (STFT)
def load_dataset_as_spectrograms_small(num_audio_samples_train=100, num_audio_samples_test=100):
    """
    Loads training and test datasets, from TIMIT and convert into spectrogram using STFT
    Arguments:
        num_audio_samples_per_class_train: number of audio per class to load into training dataset.
        num_audio_samples_test: total number of audio samples to load into training dataset.
    """

    # list initialization
    X_train, X_train_pad = [], []
    X_test, X_test_pad = [], []

    # data parsing
    while len(X_train) < num_audio_samples_train:
        sample = TRAIN_DATA.path_from_data_dir.sample()
        waveform, sample_rate = torchaudio.load_wav(os.path.join(DATA_DIR, sample.item()))
        specgram = torchaudio.transforms.MelSpectrogram(n_fft=512, win_length=10)(waveform)
        X_train.append(specgram)
    while len(X_test) < num_audio_samples_test:
        sample = TEST_DATA.path_from_data_dir.sample()
        waveform, sample_rate = torchaudio.load_wav(os.path.join(DATA_DIR, sample.item()))
        specgram = torchaudio.transforms.MelSpectrogram(n_fft=512, win_length=10)(waveform)
        X_test.append(specgram)

    # padding
    pad_to = max([len(specgram[0][0]) for specgram in X_train + X_test])
    for specgram in X_train:
        pad_by = pad_to - len(specgram[0][0])
        specgram_pad = F.pad(specgram, (0, pad_by, 0, 0), mode='constant')
        X_train_pad.append(specgram_pad.numpy())
    for specgram in X_test:
        pad_by = pad_to - len(specgram[0][0])
        specgram_pad = F.pad(specgram, (0, pad_by, 0, 0), mode='constant')
        X_test_pad.append(specgram_pad.numpy())

    # Return train and test data as numpy arrays.
    return np.array(X_train_pad), np.array(X_test_pad)

# Load dataset
# TRAINING_DATASET, TEST_DATASET = load_dataset_as_spectrograms_small()
TRAINING_DATASET, TEST_DATASET = load_dataset_as_spectrograms_small()
print('Training dataset shape:', TRAINING_DATASET.shape)
print('Test dataset shape:', TEST_DATASET.shape)
