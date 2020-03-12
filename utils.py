#!/usr/bin/env python3
# coding: utf-8

import tensorflow as tf
import numpy as np
import os
import constants


def pad(
        dataset=[],
        padding_mode='CONSTANT'):
    return pad_to(max([specgram.shape[1] for specgram in dataset]),
                  dataset=dataset, padding_mode=padding_mode)


def pad_to(
        length,
        dataset=[],
        padding_mode='CONSTANT'):
    padded_specgrams = []
    for specgram in dataset:
        padded_specgrams.append(pad_single(
            specgram, length, padding_mode=padding_mode))
    return padded_specgrams


def pad_single(
        specgram,
        length,
        padding_mode='CONSTANT'):
    pad_by = length - specgram.shape[1]
    paddings = tf.constant([[0, 0], [0, pad_by], [0, 0]])
    specgram_pad = tf.pad(specgram, paddings, padding_mode)
    return specgram_pad.numpy()


def convert_wav_to_mel_spec(
        path_to_wav,
        num_mel_filters=constants.num_mel_filters,
        lower_edge_hertz=constants.lower_edge_hertz,
        upper_edge_hertz=8000.0,
        frame_length=1024,
        frame_step=256,
        fft_length=1024):
    # extract audio and sample rate from WAV file
    raw_audio = tf.io.read_file(path_to_wav)
    audio, sample_rate = tf.audio.decode_wav(raw_audio)

    # reshape the signal to the shape of (batch_size, samples])
    signals = tf.reshape(audio, [1, -1])

    # a 1024-point STFT with frames of 64 ms and 75% overlap
    stfts = tf.signal.stft(signals, frame_length=frame_length,
                           frame_step=frame_step, fft_length=fft_length)
    spectrograms = tf.abs(stfts)

    # warp the linear scale spectrograms into the mel-scale
    num_spectrogram_bins = stfts.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_filters, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
        spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(
        spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    return mel_spectrograms


def convert_mel_spec_to_wav(
        specgrams,
        sample_rate=16000,
        num_mel_filters=32,
        lower_edge_hertz=20.0,
        upper_edge_hertz=8000.0,
        frame_length=1024,
        frame_step=256,
        fft_length=1024):
    # unwrap the mel-scale spectrogram into the linear scale
    num_spectrogram_bins = specgrams.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_filters, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)

    mel_to_linear_weight_matrix = tf.linalg.inv(linear_to_mel_weight_matrix)

    inverse_stft = tf.signal.inverse_stft(
        mel_to_linear_weight_matrix, frame_length, frame_step)

    return tf.audio.encode_wav(inverse_stft, sample_rate)


def load_dataset_mel_spectogram(
        dataset=[],
        num_audio_files=100,
        num_mel_filters=32,
        data_dir="data",
        lower_edge_hertz=20.0,
        upper_edge_hertz=8000.0):
    """
    Loads training and test datasets, from TIMIT and convert into spectrogram using STFT
    Arguments:
        num_audio_samples_per_class_train: number of audio per class to load into training dataset
    """

    # list initialization
    numpy_specgrams = []

    # data parsing
    while len(numpy_specgrams) < num_audio_files:
        sample = dataset.path_from_data_dir.sample()
        mel_specgram = convert_wav_to_mel_spec(os.path.join(
            data_dir, sample.item()), num_mel_filters=num_mel_filters)
        numpy_specgrams.append(mel_specgram)

    return np.array(pad(numpy_specgrams))
