#!/usr/bin/env python3
# coding: utf-8

import argparse
import logging as log
import model
import constants
import utils
import numpy as np
import tensorflow as tf
import shutil
import os
import librosa
import librosa.display
import librosa.feature
import matplotlib.pyplot as plt

log.basicConfig(format='%(asctime)s.%(msecs)06d: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S', level=log.INFO)

# construct argument parser
parser = argparse.ArgumentParser(
    description="Load a audio steganorgaphy tensorflow model and hide a secret message inside a cover message")
parser.add_argument("--model", required=True, help="path to trained model")
parser.add_argument("--secret", required=True,
                    help="path to secret audio file")
parser.add_argument("--cover", required=True, help="path to audio file")
parser.add_argument("--length", required=False, type=int,
                    help="length of the audio in seconds")
parser.add_argument('--plot', '-p', required=True,
                    help='plot the mel spectrograms of input and ouput audio', action='store_true', default=True)
args = vars(parser.parse_args())

# config
shape = (1, args['length'], constants.num_mel_filters)
output_dir = os.path.join('predictions', os.path.basename(args['model']))

# load model
mdl = model.steg_model(shape, pretrain=False)
mdl.load_weights(args['model'])

# convert wav to spectrograms
secret_in = utils.convert_wav_to_mel_spec(args['secret'])
cover_in = utils.convert_wav_to_mel_spec(args['cover'])

if args['length'] is not None:
    secret_in = utils.pad_single(secret_in, args['length'])
    cover_in = utils.pad_single(cover_in, args['length'])

# predict the output return two tensor wav
secret_out, cover_out = mdl.predict(
    [np.array([secret_in]), np.array([cover_in])])

# build output dir
os.makedirs(output_dir, exist_ok=True)
shutil.copyfile(args['cover'], os.path.join(
    output_dir, 'input_cover_' + os.path.basename(args['cover'])), follow_symlinks=True)
shutil.copyfile(args['secret'], os.path.join(
    output_dir, 'input_secret_' + os.path.basename(args['secret'])), follow_symlinks=True)

# convert output spectrograms to wav
for output in [
    {
        'specgram': cover_out,
        'fname': 'output_cover_' + os.path.basename(args['cover'])
    }, {
        'specgram': secret_out,
        'fname': 'output_secret_' + os.path.basename(args['secret'])
    }
]:
    print(output['specgram'].shape)
    wav = utils.convert_mel_spec_to_wav(output['specgram'][0])
    full_fname = os.path.join(output_dir, output['fname'])
    tf.io.write_file(full_fname, wav, name=None)
    log.info('Spectrogram converted to wav: {}'.format(full_fname))

if args['plot']:
    # create plot of spectrograms

    # print()
    cover_in_specgram = utils.tf_melspec_to_librosa(cover_in)
    cover_in_melspec = librosa.power_to_db(cover_in_specgram, ref=np.max)

    secret_in_specgram = utils.tf_melspec_to_librosa(secret_in)
    secret_in_melspec = librosa.power_to_db(secret_in_specgram, ref=np.max)

    cover_out_specgram = utils.tf_melspec_to_librosa(cover_out[0])
    cover_out_melspec = librosa.power_to_db(cover_out_specgram, ref=np.max)

    secret_out_specgram = utils.tf_melspec_to_librosa(secret_out[0])
    secret_out_melspec = librosa.power_to_db(secret_out_specgram, ref=np.max)

    print(cover_in_specgram.shape)
    print(secret_in_specgram.shape)
    print(cover_out_specgram.shape)
    print(secret_out_specgram.shape)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    librosa.display.specshow(cover_in_melspec, sr=constants.sample_rate,
                             hop_length=constants.frame_step, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Input cover')

    plt.subplot(2, 2, 2)
    librosa.display.specshow(cover_out_melspec, sr=constants.sample_rate,
                             hop_length=constants.frame_step, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Ouput cover')

    plt.subplot(2, 2, 3)
    librosa.display.specshow(secret_in_melspec, sr=constants.sample_rate,
                             hop_length=constants.frame_step, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Input secret')

    plt.subplot(2, 2, 4)
    librosa.display.specshow(secret_out_melspec, sr=constants.sample_rate,
                             hop_length=constants.frame_step, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Output secret')

    # plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot'))
