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
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# import TIMIT dataset
data_dir = "data"
train_csv = pd.concat([pd.read_csv(os.path.join(
    data_dir, "train_data.csv")), pd.read_csv(os.path.join(data_dir, "test_data.csv"))])
train_data = [sample for sample in train_csv[train_csv.path_from_data_dir.str.contains(
    'WAV.wav', na=False)]['path_from_data_dir']]
del train_csv

# configuration statistics
log.info('Training examples: {}'.format(len(train_data)))

# parse command line args
parser = argparse.ArgumentParser(
    description='Train a model to hide a secret audio message into a cover audio message'
)
parser.add_argument(
    '--epochs', '-e', default=constants.epochs, help='number of epochs')
parser.add_argument('--samples', '-s', default=constants.num_samples,
                    help='number of sample to train the model with')
parser.add_argument('--batchSize', '-b',
                    default=constants.batch_size, help='training batch size')
parser.add_argument('--frameLength', '-f', default=constants.frame_length,
                    help='length of the stft window in frames')
parser.add_argument('--saveDataset', '-sD', action='store_true', default=True,
                    help='serialize dataset once parsed')
parser.add_argument('--loadDataset', '-lD', type=str,
                    help='parse serialized dataset')
parser.add_argument('--fixedDataset', '-fD', action='store_true', default=False,
                    help='dataset has fixed length')
args = vars(parser.parse_args())

# validate input params
if args['samples'] > len(train_data) or constants.num_samples > len(train_data):
    sys.exit('Error: there are only {} samples in the dataset, use a smaller sample size'.format(
        len(train_data)))

if args['loadDataset'] is not None:
    x_train = np.load(args['loadDataset'])
    log.info('Dataset loaded from {}'.format(args['loadDataset']))
else:
    x_train = utils.load_dataset_mel_spectrogram(
        dataset=train_data,
        data_dir=data_dir,
        num_audio_files=args['samples'],
        frame_length=args['frameLength'],
        fixed_length=args['fixedDataset'])
    if args['saveDataset']:
        datasetFname = 'dataset-{}-{}-f{}'.format(
            datetime.datetime.now().strftime("%Y%m%d_%H%M"),
            args['samples'], args['frameLength'])
        np.save(datasetFname, x_train)
        log.info('Dataset saved into {}.npy'.format(datasetFname))

log.info('Samples shape: {}'.format(x_train.shape))
del train_data

# we split training set into two halfs.
train_spectrograms_shape = x_train.shape
log.info('Generating model instance')
model = model.steg_model(train_spectrograms_shape[1:], pretrain=False)

log.info('Generating secret training data subset')
secret_audio_files = x_train[0:train_spectrograms_shape[0] // 2]
log.info('Generating cover training data subset')
cover_audio_files = x_train[train_spectrograms_shape[0] // 2:]
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
callback_tensorboard = TensorBoard(
    log_dir=log_dir, histogram_freq=1)
callback_checkpoint = ModelCheckpoint(
    log_dir, monitor='loss', verbose=1, save_best_only=True, mode='max')
callback_lr_schedule = LearningRateScheduler(lr_scheduler)

data_gen = generator.DataGenerator(x_data, samples=args['samples'],
                                   batch_size=args['batchSize'])

# train model
model.fit(x=data_gen,
          epochs=args['epochs'],
          steps_per_epoch=math.floor(args['samples'] // args['batchSize']),
          #   use_multiprocessing=True,
          callbacks=[callback_lr_schedule, callback_tensorboard, callback_checkpoint])

# save model
model_hdf5 = '{}-{}-n{}.hdf5'.format(
    socket.gethostname(),
    datetime.datetime.now().strftime("%Y%m%d_%H%M"),
    train_spectrograms_shape[2])
model.save_weights(model_hdf5)
log.info('Model weights saved at {}'.format(model_hdf5))
