#!/usr/bin/env python3
# coding: utf-8

import logging as log
import utils
import tensorflow as tf
import shutil
import os

log.basicConfig(format='%(asctime)s.%(msecs)06d: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S', level=log.INFO)

# config
output_dir = 'test_transform'
data_file = 'data/TRAIN/DR1/FCJF0/SA1.WAV.wav'

os.makedirs(output_dir, exist_ok=True)
specgram = utils.convert_wav_to_mel_spec(data_file)
shutil.copyfile(data_file, os.path.join(
    output_dir, 'input_' + os.path.basename(data_file)), follow_symlinks=True)
wav = utils.convert_mel_spec_to_wav(specgram)
full_fname = os.path.join(output_dir, 'output_' + os.path.basename(data_file))
tf.io.write_file(full_fname, wav, name=None)
log.info('Spectrogram converted to wav: {}'.format(full_fname))
