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
        upper_edge_hertz=constants.upper_edge_hertz,
        frame_length=constants.frame_length,
        frame_step=constants.frame_step,
        fft_length=constants.fft_length):

    # extract audio and sample rate from WAV file
    raw_audio = tf.io.read_file(path_to_wav)
    audio, sample_rate = tf.audio.decode_wav(raw_audio)
    print('audio', audio.shape)
    print('sample_rate', sample_rate)

    # reshape the signal to the shape of (batch_size, samples])
    signals = tf.linalg.pinv(audio)
    print('signals', signals.shape)

    # a 1024-point STFT with frames of 64 ms and 75% overlap
    stfts = tf.signal.stft(
        signals,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length)
    print('stfts', stfts.shape)
    spectrograms = tf.abs(stfts)
    print('spectrograms', spectrograms.shape)

    # warp the linear scale spectrograms into the mel-scale
    num_spectrogram_bins = stfts.shape[-1]
    print('num_spectrogram_bins', num_spectrogram_bins)
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_filters,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sample_rate,
        lower_edge_hertz=lower_edge_hertz,
        upper_edge_hertz=upper_edge_hertz
    )
    print('linear_to_mel_weight_matrix', linear_to_mel_weight_matrix.shape)
    mel_spectrograms = tf.tensordot(
        spectrograms, linear_to_mel_weight_matrix, 1)
    print('mel_spectrograms', mel_spectrograms.shape)
    mel_spectrograms.set_shape(
        spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    print('mel_spectrograms', mel_spectrograms.shape)
    print('---------------------------')

    return mel_spectrograms


def convert_mel_spec_to_wav(
        specgrams,
        sample_rate=constants.sample_rate,
        num_mel_filters=constants.num_mel_filters,
        lower_edge_hertz=constants.lower_edge_hertz,
        upper_edge_hertz=constants.upper_edge_hertz,
        frame_length=constants.frame_length,
        frame_step=constants.frame_step,
        fft_length=constants.fft_length):
    specgrams = tf.convert_to_tensor(specgrams)

    num_spectrogram_bins = 513
    print('num_spectrogram_bins', num_spectrogram_bins)

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_filters,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sample_rate,
        lower_edge_hertz=lower_edge_hertz,
        upper_edge_hertz=upper_edge_hertz
    )
    print('linear_to_mel_weight_matrix', linear_to_mel_weight_matrix.shape)

    mel_to_linear_weight_matrix = tf.linalg.pinv(linear_to_mel_weight_matrix)
    print('mel_to_linear_weight_matrix', mel_to_linear_weight_matrix.shape)

    stfts = tf.tensordot(specgrams, mel_to_linear_weight_matrix, axes=1)
    print('stfts', stfts.shape)
    stfts.set_shape(
        specgrams.shape[:-1].concatenate(mel_to_linear_weight_matrix.shape[-1:]))
    print('stfts', stfts.shape)

    inverse_stft = tf.signal.inverse_stft(
        stfts=tf.complex(stfts, 0.0),
        frame_length=frame_length,
        frame_step=frame_step
    )
    print('inverse_stft', inverse_stft.shape)

    signals = tf.linalg.pinv(inverse_stft)
    print('signals', signals.shape)

    return tf.audio.encode_wav(signals, sample_rate)


def load_dataset_mel_spectogram(
        dataset=[],
        data_dir="data",
        num_audio_files=100,
        num_mel_filters=constants.num_mel_filters,
        lower_edge_hertz=constants.lower_edge_hertz,
        upper_edge_hertz=constants.upper_edge_hertz):
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
