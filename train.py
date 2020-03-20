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

# tensorflow
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

# memory investigation
from pympler import muppy, summary

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
parser.add_argument('--melFilters', '-m', default=constants.num_mel_filters,
                    help='number of mel filters to apply')
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
        num_mel_filters=args['melFilters'],
        fixed_length=args['fixedDataset'])
    if args['saveDataset']:
        datasetFname = 'dataset-{}'.format(
            datetime.datetime.now().strftime("%Y%m%d_%H%M"))
        np.save(datasetFname, x_train)
        log.info('Dataset saved into {}.npy'.format(datasetFname))

log.info('Samples shape: {}'.format(x_train.shape))
del train_data

# we split training set into two halfs.
train_spectrograms = x_train.shape[2]
secret_audio_files = x_train[0:x_train.shape[0] // 2]
cover_audio_files = x_train[x_train.shape[0] // 2:]
del x_train

summary.print_(summary.summarize(muppy.get_objects()))

model = model.steg_model(cover_audio_files.shape[1:], pretrain=False)

x_data = [secret_audio_files, cover_audio_files]
y_data = np.concatenate((secret_audio_files, cover_audio_files), axis=3)

# callbacks
callback_tensorboard = TensorBoard(
    log_dir=log_dir, histogram_freq=1)
callback_checkpoint = ModelCheckpoint(
    log_dir, monitor='loss', verbose=1, save_best_only=True, mode='max')

# train model
model.fit(x=x_data, y=x_data, epochs=args['epochs'],
          batch_size=args['batchSize'], callbacks=[callback_tensorboard, callback_checkpoint])

# save model
model_hdf5 = 'model-{}-n{}.hdf5'.format(
    datetime.datetime.now().strftime("%Y%m%d_%H%M"), train_spectrograms)
model.save_weights(model_hdf5)
log.info('Model weights saved at {}'.format(model_hdf5))
