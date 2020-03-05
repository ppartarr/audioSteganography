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

# Print statistics.
print("Number of training examples = " + str(TRAIN_DATA.shape[0]))
print("Number of test examples = " + str(TEST_DATA.shape[0]))
print("Training data shape: " + str(TRAIN_DATA.shape)) # Should be (train_size, 64, 64, 3).

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
train_data, test_data = load_dataset_as_spectrograms_small()
print(train_data.shape)
print(test_data.shape)


# We split training set into two halfs.
secret_audio_files = train_data[0:train_data.shape // 2]

# # C: cover audio
cover_audio_files = train_data[train_data.shape // 2:]


# Create the encoder and decoder networks
class CoverEncoderNet(nn.Module):

    def __init__(self):
        """
        In this constructor we instantiate a 3 layer neural network with two lin
        """
        super(CoverEncoderNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        return x

class CoverDecoderNet(nn.Module):

    def __init__(self):
        """
        In this constructor we instantiate a 3 layer neural network with two lin
        """
        super(CoverDecoderNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        return x

class SecretDecoderNet(nn.Module):

    def __init__(self):
        """
        In this constructor we instantiate a 3 layer neural network with two lin
        """
        super(SecretDecoderNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        return x


cover_encoder = CoverEncoderNet()
cover_decoder = CoverDecoderNet()
secret_decoder = SecretDecoderNet()


carrier = cover_encoder(train_data)
modified_cover = cover_decoder(carrier)
modified_secret = secret_decoder(modified_cover)



# Use ADAM as an optimizer
# We use the default learning rate (lr) of 1e-3
# TODO test with different weight decays
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)

def train(model, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        # TODO run on GPU for training
        # data = data.to(device)
        # target = target.to(device)
        data = data.requires_grad_() #set requires_grad to True for training
        output = model(data)
        output = output.permute(1, 0, 2) #original output dimensions are batchSizex1x10
        loss = F.nll_loss(output[0], target) #the loss functions expects a batchSizex10 input
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0: #print training stats
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))
