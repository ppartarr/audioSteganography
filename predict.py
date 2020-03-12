#!/usr/bin/env python3
# coding: utf-8

import argparse
import logging as log
import model
import constants
import utils
import numpy as np
import tensorflow as tf

log.basicConfig(format='%(asctime)s.%(msecs)06d: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S', level=log.INFO)

# construct argument parser
parser = argparse.ArgumentParser(
    description="Load a audio steganorgaphy tensorflow model and hide a secret message inside a cover message")
parser.add_argument("--model", required=True, help="path to trained model")
parser.add_argument("--secret", required=True,
                    help="path to secret audio file")
parser.add_argument("--cover", required=True, help="path to audio file")
parser.add_argument("--length", required=True,
                    help="length of the spectrogram", type=int)
args = vars(parser.parse_args())

# load model
print(args['length'])
shape = (1, args['length'], constants.num_mel_filters)
mdl = model.steg_model(shape, pretrain=False)
mdl.load_weights(args['model'])

secret_audio = utils.convert_wav_to_mel_spec(args['secret_audio'])
cover_audio = utils.convert_wav_to_mel_spec(args['cover_audio'])

print(np.array([secret_audio]).shape)
print(secret_audio.shape)
print(cover_audio.shape)

dataset = [secret_audio, cover_audio]
secret_audio = utils.pad_single(secret_audio, args['length'])
cover_audio = utils.pad_single(cover_audio, args['length'])

# predict the output
secret_out, cover_out = mdl.predict(
    [np.array([secret_audio]), np.array([cover_audio])])

print(type(secret_out))
wav = utils.convert_mel_spec_to_wav(secret_out)
audio = tf.audio.decode_wav(wav)

# play audio
# IPython.display.Audio(wav)
# IPython.display.Audio(secret_in)
# IPython.display.Audio(cover_out)
# IPython.display.Audio(secret_out)
