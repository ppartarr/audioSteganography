#!/usr/bin/env python3
# coding: utf-8

import logging as log
import utils
import tensorflow as tf
import shutil
import os
import librosa
import librosa.display
import constants
import numpy as np
import matplotlib.pyplot as plt

log.basicConfig(format='%(asctime)s.%(msecs)06d: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S', level=log.INFO)

# config
output_dir = os.path.join(constants.data_dir, 'transform')
data_file = os.path.join(constants.data_dir, 'TRAIN/DR1/FCJF0/SA1.WAV.wav')

os.makedirs(output_dir, exist_ok=True)

# write input file
shutil.copyfile(data_file, os.path.join(
    output_dir, 'input_' + os.path.basename(data_file)), follow_symlinks=True)

full_fname = os.path.join(output_dir, 'output_' + os.path.basename(data_file))

# convert using librosa
specgram = utils.convert_wav_to_stft_spec(data_file)
wav = utils.convert_stft_spec_to_wav(specgram)
tf.io.write_file(full_fname, wav, name=None)

# show input & output spectrograms
input_specgram = utils.tf_melspec_to_librosa(specgram)
input_decibel = librosa.power_to_db(input_specgram, ref=np.max)

output_specgram = utils.tf_melspec_to_librosa(
    utils.convert_wav_to_stft_spec(full_fname))
output_decibel = librosa.power_to_db(output_specgram, ref=np.max)

# save secret & cover spectrograms
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
librosa.display.specshow(input_decibel, sr=constants.sample_rate,
                         hop_length=constants.hop_length, y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Input wav file as mel spec')

plt.subplot(2, 1, 2)
librosa.display.specshow(output_decibel, sr=constants.sample_rate,
                         hop_length=constants.hop_length, y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Ouput wav file as mel spec')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plot'))
