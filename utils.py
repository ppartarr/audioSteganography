#!/usr/bin/env python3
# coding: utf-8

import tensorflow as tf
import numpy as np
import os
import constants
import librosa


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


def convert_wav_to_mel_spec(
        path_to_wav,
        sr=constants.sample_rate,
        n_mels=constants.num_mel_filters,
        fmax=constants.upper_edge_hertz,
        hop_length=constants.frame_step,
        sample_rate=constants.sample_rate,
        n_fft=constants.num_fft):

    audio, sample_rate = librosa.load(path_to_wav, sr=constants.sample_rate)

    print('wav input shape: ', audio.shape)

    librosa_melspec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmax=fmax,
        center=True)

    # convert mel spectogram representation from librosa to tensorflow
    tf_melspec = tf.expand_dims(
        tf.transpose(tf.convert_to_tensor(librosa_melspec)),
        0
    )
    print('librosa melspec shape: ', tf_melspec.shape)
    print('---------------------------')
    return tf_melspec


def convert_wav_to_mel_spec_librosa(
        path_to_wav,
        sr=constants.sample_rate,
        n_mels=constants.num_mel_filters,
        fmax=constants.upper_edge_hertz,
        hop_length=constants.frame_step,
        sample_rate=constants.sample_rate,
        n_fft=constants.num_fft):

    audio, sample_rate = librosa.load(path_to_wav, sr=constants.sample_rate)

    print('librosa wav input shape: ', audio.shape)

    librosa_melspec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        fmax=fmax,
        n_mels=n_mels)

    return librosa_melspec


def convert_mel_spec_to_wav_librosa(
        tf_melspec,
        sr=constants.sample_rate,
        n_mels=constants.num_mel_filters,
        fmax=constants.upper_edge_hertz,
        hop_length=constants.frame_step,
        sample_rate=constants.sample_rate,
        n_fft=constants.num_fft):

    print('librosa mel spec input shape: ', tf_melspec.shape)

    # convert mel spectogram representation from tensorflow to librosa
    librosa_melspec = tf.transpose(tf_melspec[0])

    librosa_wav = librosa.feature.inverse.mel_to_audio(
        librosa_melspec.numpy(),
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        fmax=fmax)

    # convert librosa wav back to tensor wav
    return librosa_wav


def convert_mel_spec_to_wav(
        tf_melspec,
        sr=constants.sample_rate,
        n_mels=constants.num_mel_filters,
        fmax=constants.upper_edge_hertz,
        hop_length=constants.frame_step,
        sample_rate=constants.sample_rate,
        n_fft=constants.num_fft):

    print('librosa mel spec input shape: ', tf_melspec.shape)

    # convert mel spectogram representation from tensorflow to librosa
    librosa_melspec = tf.transpose(tf_melspec[0])

    librosa_wav = librosa.feature.inverse.mel_to_audio(
        librosa_melspec.numpy(),
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        fmax=fmax,
        center=True)

    # convert librosa wav back to tensor wav
    return librosa_wav_to_tf(librosa_wav)


def tf_melspec_to_librosa(tf_melspec):
    return tf.transpose(tf_melspec[0])


def librosa_melspec_to_rf(librosa_melspec):
    return tf.expand_dims(
        tf.transpose(tf.convert_to_tensor(librosa_melspec)),
        0
    )


def librosa_wav_to_tf(wav, sample_rate=constants.sample_rate):
    audio = tf.transpose(
        tf.expand_dims(
            tf.convert_to_tensor(wav),
            0
        )
    )

    return tf.audio.encode_wav(audio, sample_rate)
