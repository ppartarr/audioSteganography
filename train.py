#!/usr/bin/env python3
# coding: utf-8

import argparse
import datetime
import numpy as np
import os
import pandas as pd
import utils
import model
import constants
import sys
import socket
import generator
import math

# tensorflow
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler

# logging
import logging as log
log.basicConfig(format='%(asctime)s.%(msecs)06d: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S', level=log.INFO)

# parse command line args
parser = argparse.ArgumentParser(
    description='Train a model to hide a secret audio message into a cover audio message'
)
parser.add_argument(
    '--epochs', '-e', type=int, default=constants.epochs, help='Number of epochs')
parser.add_argument(
    '--samples', '-s', type=int, default=constants.num_samples, help='Number of sample to train the model with')
parser.add_argument(
    '--batch', '-b', type=int, default=constants.batch_size, help='Training batch size')
parser.add_argument(
    '--fft', '-f', type=int, default=constants.n_fft, help='Number of FFT(s)')
parser.add_argument(
    '--datadir', '-d', type=str, default=constants.data_dir, help='Data directory')
parser.add_argument(
    '--fixedDataset', '-fd', action='store_true', default=False, help='Dataset has fixed length')
args = vars(parser.parse_args())

# import TIMIT dataset
train_csv = pd.concat([pd.read_csv(os.path.join(
    args['datadir'], 'train_data.csv')), pd.read_csv(os.path.join(args['datadir'], 'test_data.csv'))])
train_data = [sample for sample in train_csv[train_csv.path_from_data_dir.str.contains(
    'WAV.wav', na=False)]['path_from_data_dir']]
del train_csv

# validate input params
if args['samples'] > len(train_data) or constants.num_samples > len(train_data):
    sys.exit('Error: there are only {} samples in the dataset, use a smaller sample size'.format(
        len(train_data)))

x_train = utils.load_dataset_stft_spectrogram(
    dataset=train_data,
    data_dir=args['datadir'],
    num_samples=args['samples'],
    n_fft=args['fft'],
    fixed_length=args['fixedDataset'])

del train_data

# configuration statistics
log.info('Training examples: {}'.format(len(x_train)))
log.info('Samples shape: {}'.format(x_train.shape))

# we split training set into two halfs.
train_spectrograms_shape = x_train.shape
log.info('Generating model instance')
model = model.steg_model(train_spectrograms_shape[1:], pretrain=False)

log.info('Generating secret training data subset')
secret_audio_files = x_train
log.info('Generating cover training data subset')
cover_audio_files = np.flip(x_train)
del x_train

log.info('Generating data given to train function')
x_data = [secret_audio_files, cover_audio_files]
del secret_audio_files
del cover_audio_files


def lr_scheduler(epoch):
    if epoch < 200:
        return 0.001
    elif epoch < 400:
        return 0.0003
    elif epoch < 600:
        return 0.0001
    else:
        return 0.00003


# callbacks
log_dir = os.path.join(
    args['datadir'], 'fit', datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
)
callback_lr_schedule = LearningRateScheduler(lr_scheduler)
callback_tensorboard = TensorBoard(
    log_dir,
    histogram_freq=1)
callback_checkpoint = ModelCheckpoint(
    log_dir,
    monitor='loss',
    verbose=1,
    save_best_only=True,
    mode='max')

# data generator
data_gen = generator.DataGenerator(
    x_data, samples=args['samples'], batch_size=args['batch'])

# train model
model.fit(
    x=data_gen,
    epochs=args['epochs'],
    steps_per_epoch=math.floor(args['samples'] // args['batch']),
    callbacks=[
        callback_lr_schedule,
        callback_tensorboard,
        callback_checkpoint
    ])

# save model
model_dir = os.path.join(args['datadir'], 'models')
os.makedirs(model_dir, exist_ok=True)
model_hdf5 = '{}-{}-n{}.hdf5'.format(
    socket.gethostname(),
    datetime.datetime.now().strftime('%Y%m%d_%H%M'),
    train_spectrograms_shape[2])
model_path = os.path.join(model_dir, model_hdf5)
model.save_weights(model_path)
log.info('Model weights saved at {}'.format(model_path))
