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


def load_dataset_mel_spectrogram(
        dataset=[],
        data_dir="data",
        num_audio_files=100,
        num_mel_filters=constants.num_mel_filters,
        lower_edge_hertz=constants.lower_edge_hertz,
        upper_edge_hertz=constants.upper_edge_hertz,
        fixed_length=False):
    """
    Loads training and test datasets, from TIMIT and convert into spectrogram using STFT
    Arguments:
        num_audio_samples_per_class_train: number of audio per class to load into training dataset
    """

    if fixed_length:
        return load_fixed_dataset_mel_spectrogram(
            dataset, data_dir, num_audio_files,
            num_mel_filters, lower_edge_hertz, upper_edge_hertz)

    # list initialization
    numpy_specgrams = None

    # padding vars
    longest_specgram = 0

    # data parsing
    for sample in dataset:
        if numpy_specgrams is not None and len(numpy_specgrams) == num_audio_files:
            break

        mel_specgram = convert_wav_to_mel_spec(os.path.join(
            data_dir, sample), num_mel_filters=num_mel_filters)

        if numpy_specgrams is None:
            numpy_specgrams = mel_specgram[np.newaxis, ...]
            longest_specgram = mel_specgram.shape[1]
        else:
            if longest_specgram < mel_specgram.shape[1]:
                # pad parsed specgrams
                pad_by = mel_specgram.shape[1] - longest_specgram
                padding = tf.constant([[0, 0], [0, 0], [0, pad_by], [0, 0]])
                numpy_specgrams = tf.pad(numpy_specgrams, padding, 'CONSTANT')
                longest_specgram = mel_specgram.shape[1]
            elif longest_specgram > mel_specgram.shape[1]:
                # pad new specgram
                pad_by = longest_specgram - mel_specgram.shape[1]
                padding = tf.constant([[0, 0], [0, pad_by], [0, 0]])
                mel_specgram = tf.pad(mel_specgram, padding, 'CONSTANT')

            numpy_specgrams = np.concatenate(
                (numpy_specgrams, mel_specgram[np.newaxis, ...]), axis=0)

        print('Parsing data progress: {}% ({}/{})'.format(
            len(numpy_specgrams) * 100 // num_audio_files, len(numpy_specgrams), num_audio_files), end="\r")

    return numpy_specgrams


def load_fixed_dataset_mel_spectrogram(
        dataset=[],
        data_dir="data",
        num_audio_files=100,
        num_mel_filters=constants.num_mel_filters,
        lower_edge_hertz=constants.lower_edge_hertz,
        upper_edge_hertz=constants.upper_edge_hertz,
        fixed_length=False):

    # list initialization
    sample_specgram = convert_wav_to_mel_spec(os.path.join(
        data_dir, dataset[0]), num_mel_filters=num_mel_filters)
    numpy_specgrams = np.empty(
        (num_audio_files, sample_specgram.shape[0], sample_specgram.shape[1], sample_specgram.shape[2]), dtype=np.float32)
    numpy_specgrams.flags.writeable = True

    # data parsing
    for idx in range(num_audio_files):
        sample = dataset[idx]

        mel_specgram = convert_wav_to_mel_spec(os.path.join(
            data_dir, sample), num_mel_filters=num_mel_filters)

        numpy_specgrams[idx] = mel_specgram

        print('Parsing data progress: {}% ({}/{})'.format(
            (idx + 1) * 100 // num_audio_files, idx + 1, num_audio_files), end="\r")

    return numpy_specgrams


def convert_wav_to_mel_spec(
        path_to_wav,
        sr=constants.sample_rate,
        num_mel_filters=constants.num_mel_filters,
        fmax=constants.upper_edge_hertz,
        hop_length=constants.frame_step,
        sample_rate=constants.sample_rate,
        n_fft=constants.frame_length):
    """
    Converts a raw wav to a Tensor mel spectrogram
        Raw wave shape: (samples, 1)
        Tensor mel spectrogram shape: (1, t, num_mel_filters)
    """

    audio, sample_rate = librosa.load(path_to_wav, sr=constants.sample_rate)

    librosa_melspec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=num_mel_filters,
        fmax=fmax,
        center=True)

    return librosa_melspec_to_tf(librosa_melspec)


def convert_mel_spec_to_wav(
        tf_melspec,
        sr=constants.sample_rate,
        fmax=constants.upper_edge_hertz,
        hop_length=constants.frame_step,
        sample_rate=constants.sample_rate,
        n_fft=constants.frame_length):
    """
    Converts a Tensor mel spectrogram to a Tensor wav
        Tensor mel spectrogram shape: (1, t, num_mel_filters)
        Tensor wav shape: (1, t)
    """

    librosa_melspec = tf_melspec_to_librosa(tf_melspec)

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
    """
    Converts a Tensorflow Tensor mel spectrogram to a Librosa mel spectrogram
        Tensor mel spectrogram shape: (1, t, num_mel_filters)
        Librosa mel spectrogram shape: (num_mel_filters, t)
    """
    return tf.transpose(tf_melspec[0])


def librosa_melspec_to_tf(librosa_melspec):
    """
    Converts a tensorflow Librosa mel spectrogram to a Tensorflow mel spectrogram
        Tensor mel spectrogram shape: (1, t, num_mel_filters)
        Librosa mel spectrogram shape: (num_mel_filters, t)
    """
    return tf.expand_dims(
        tf.transpose(tf.convert_to_tensor(librosa_melspec)),
        0
    )


def librosa_wav_to_tf(librosa_wav, sample_rate=constants.sample_rate):
    """
    Converts a Librosa wav file to a Tensorflow Tensor
        Librosa wav shape: t
        Tensorflow wav shape: (1, t)
    """
    audio = tf.transpose(
        tf.expand_dims(
            tf.convert_to_tensor(librosa_wav),
            0
        )
    )

    return tf.audio.encode_wav(audio, sample_rate)


def tf_wav_to_librosa(tf_wav, sample_rate=constants.sample_rate):
    """
    Converts a Tensorflow Tensor wav file to a Librosa wav
        Librosa wav shape: t
        Tensorflow wav shape: (1, t)
    """
    # print(tf_wav.shape)
    # print(tf_wav[0])
    audio, sample_rate = tf.audio.decode_wav(tf_wav)
    print('audio shape: ', audio.shape)
    return audio[0]


def calculate_num_mels_from_audio_length(
        frame_step=constants.frame_step,
        sample_rate=constants.sample_rate,
        audio_length=constants.audio_length):

    return (audio_length * sample_rate) / frame_step
