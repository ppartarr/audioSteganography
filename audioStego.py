#!/usr/bin/env python
# coding: utf-8




# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# From https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import matplotlib.pyplot as plt
import random
import math
import IPython





# Import data from TIMIT dataset
TEST_DATA = pd.read_csv("./data/test_data.csv")
TRAIN_DATA = pd.read_csv("./data/train_data.csv")
DATA_DIR = "./data/"
print(TRAIN_DATA.path_from_data_dir)





# Print statistics.
print("Number of training examples = " + str(TRAIN_DATA.shape[0]))
print("Number of test examples = " + str(TEST_DATA.shape[0]))
print("Training data shape: " + str(TRAIN_DATA.shape)) # Should be (train_size, 64, 64, 3).





# Convert audio to spectrogram using Short Time Fourier Transform (STFT)
def load_dataset_as_spectrograms_small(num_audio_samples_per_class_train=10, num_audio_samples_test=500):
    """Loads training and test datasets, from TIMIT and convert into spectrogram using STFT

    Arguments:
        num_audio_samples_per_class_train: number of audio per class to load into training dataset.
        num_audio_samples_test: total number of audio samples to load into training dataset.
    """
    X_train = []
    X_test = []
    SAMPLES = 10

    # Create training set
    TRAIN_WAV = []
    for audio_sample in TRAIN_DATA.path_from_data_dir:
        if len(TRAIN_WAV) == SAMPLES:
            break
        if type(audio_sample) == str and "WAV.wav" in audio_sample:
            TRAIN_WAV.append(audio_sample)
    random.shuffle(TRAIN_WAV)

    for audio_sample_path in TRAIN_WAV:
        waveform, sample_rate = torchaudio.load_wav(DATA_DIR + audio_sample_path)
        specgram = torchaudio.transforms.MelSpectrogram(n_fft=512, win_length=10)(waveform)
        X_train.append(specgram)

    TEST_WAV = []
    for audio_sample in TEST_DATA.path_from_data_dir:
        if len(TEST_WAV) == SAMPLES:
            break
        if type(audio_sample) == str and "WAV.wav" in audio_sample:
            TEST_WAV.append(audio_sample)
    random.shuffle(TEST_WAV)

    for audio_sample_path in TEST_WAV:
        waveform, sample_rate = torchaudio.load_wav(DATA_DIR + audio_sample_path)
        specgram = torchaudio.transforms.MelSpectrogram(n_fft=512, win_length=10)(waveform)
        X_test.append(specgram)

    # padding
    max_len = 0
    for specgram in X_train + X_test:
        if len(specgram[0][0]) > max_len:
            max_len = len(specgram[0][0])
    X_train_pad = []
    X_test_pad = []
    for specgram in X_train:
        pad_by = max_len - len(specgram[0][0])
        specgram_pad = F.pad(specgram, (0, pad_by, 0, 0), mode='constant')
        X_train_pad.append(specgram_pad.numpy())
    for specgram in X_test:
        pad_by = max_len - len(specgram[0][0])
        specgram_pad = F.pad(specgram, (0, pad_by, 0, 0), mode='constant')
        X_test_pad.append(specgram_pad.numpy())

    # Return train and test data as numpy arrays.
    return np.array(X_train_pad), np.array(X_test_pad)





# Load dataset
TRAINING_DATASET, TEST_DATASET = load_dataset_as_spectrograms_small()
print(TRAINING_DATASET.shape)
print(TEST_DATASET.shape)





# We split training set into two halfs.
# First half is used for training as secret images, second half for cover images.

# S: secret image
input_S = TRAINING_DATASET[0:X_train.shape[0] // 2]

# C: cover image
input_C = TEST_DATASET[X_train.shape[0] // 2:]






#

