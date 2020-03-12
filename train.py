#!/usr/bin/env python3
# coding: utf-8

import datetime
import numpy as np
import os
import pandas as pd
import utils
import model
import constants

# tensorflow
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

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

# shape of the data is (n, 1, x, 128) where
#   n is the number of audio files
#   1 is the number of channels (mono)
#   x is the number of 64ms spectrograms with 716% overlap
#   128 is the number of mel filters
x_train = utils.load_dataset_mel_spectogram(
    dataset=train_data, num_audio_files=constants.num_samples, num_mel_filters=constants.num_mel_filters, data_dir=data_dir)
x_test = utils.load_dataset_mel_spectogram(
    dataset=test_data, num_audio_files=constants.num_samples, num_mel_filters=constants.num_mel_filters, data_dir=data_dir)
del train_csv
del train_data
del test_csv
del test_data

# we split training set into two halfs.
secret_audio_files = x_train[0:x_train.shape[0] // 2]
cover_audio_files = x_train[x_train.shape[0] // 2:]

summary.print_(summary.summarize(muppy.get_objects()))

# print(type(secret_audio_files[0]))
# print(secret_audio_files.shape[1:])
model = model.steg_model(cover_audio_files.shape[1:], pretrain=False)

x_data = [secret_audio_files, cover_audio_files]
y_data = np.concatenate((secret_audio_files, cover_audio_files), axis=3)

# callbacks
callback_tensorboard = TensorBoard(
    log_dir=log_dir, histogram_freq=1)
callback_checkpoint = ModelCheckpoint(
    log_dir, monitor='loss', verbose=1, save_best_only=True, mode='max')

# train model
model.fit(x=x_data, y=x_data, epochs=constants.epochs,
          batch_size=constants.batch_size, callbacks=[callback_tensorboard, callback_checkpoint])

# save model
model_hdf5 = 'model-{}.hdf5'.format(
    datetime.datetime.now().strftime("%Y%m%d_%H%M"))
model.save_weights(model_hdf5)
log.info('Model weights saved at {}'.format(model_hdf5))
