#!/usr/bin/env python3
# coding: utf-8

import tensorflow as tf
import numpy as np
import os
import constants
import librosa


def pad(dataset=[], padding_mode='CONSTANT'):
    return pad_to(max([specgram.shape[1] for specgram in dataset]),
                  dataset=dataset, padding_mode=padding_mode)


def pad_to(length, dataset=[], padding_mode='CONSTANT'):
    padded_specgrams = []
    for specgram in dataset:
        padded_specgrams.append(pad_single(
            specgram, length, padding_mode=padding_mode))
    return padded_specgrams


def pad_single(specgram, length, padding_mode='CONSTANT'):
    pad_by = length - specgram.shape[1]
    paddings = tf.constant([[0, 0], [0, pad_by], [0, 0]])
    specgram_pad = tf.pad(specgram, paddings, padding_mode)
    return specgram_pad.numpy()


def load_dataset_stft_spectrogram(
        dataset=[],
        data_dir="data",
        num_samples=constants.num_samples,
        n_fft=constants.n_fft,
        fmax=constants.fmax,
        fixed_length=False):
    """
    Loads training and test datasets, from TIMIT and convert into spectrogram using STFT
    Arguments:
        num_audio_samples_per_class_train: number of audio per class to load into training dataset
    """

    if fixed_length:
        return load_fixed_dataset_stft_spectrogram(
            dataset=dataset,
            data_dir=data_dir,
            n_fft=n_fft,
            num_samples=num_samples,
            fmax=fmax)

    # list initialization
    numpy_specgrams = None

    # padding vars
    longest_specgram = 0

    # data parsing
    for sample in dataset:
        if numpy_specgrams is not None and len(numpy_specgrams) == num_samples:
            break

        mel_specgram = convert_wav_to_stft_spec(os.path.join(
            data_dir, sample), n_fft=n_fft)

        if numpy_specgrams is None:
            numpy_specgrams = mel_specgram[np.newaxis, ...]
            longest_specgram = mel_specgram.shape[1]
        else:
            if longest_specgram < mel_specgram.shape[1]:
                # pad parsed specgrams
                pad_by = mel_specgram.shape[1] - longest_specgram
                numpy_specgrams = tf.pad(numpy_specgrams, tf.constant(
                    [[0, 0], [0, 0], [0, pad_by], [0, 0]]))
                longest_specgram = mel_specgram.shape[1]
            elif longest_specgram > mel_specgram.shape[1]:
                # pad new specgram
                mel_specgram = pad_single(mel_specgram, longest_specgram)

            numpy_specgrams = np.concatenate(
                (numpy_specgrams, mel_specgram[np.newaxis, ...]), axis=0)

        print('Parsing data progress: {}% ({}/{})'.format(
            len(numpy_specgrams) * 100 // num_samples, len(numpy_specgrams), num_samples), end="\r")

    return numpy_specgrams


def load_fixed_dataset_stft_spectrogram(
        dataset=[],
        data_dir="data",
        num_samples=constants.num_samples,
        n_fft=constants.n_fft,
        fmax=constants.fmax,
        fixed_length=False):

    # list initialization
    sample_specgram = convert_wav_to_stft_spec(os.path.join(
        data_dir, dataset[0]), n_fft=n_fft)
    numpy_specgrams = np.empty((
        num_samples,
        sample_specgram.shape[0],
        sample_specgram.shape[1],
        sample_specgram.shape[2]
    ), dtype=np.complex64)
    numpy_specgrams.flags.writeable = True

    # data parsing
    for idx in range(num_samples):
        sample = dataset[idx]

        mel_specgram = convert_wav_to_stft_spec(os.path.join(
            data_dir, sample))

        numpy_specgrams[idx] = mel_specgram

        print('Parsing data progress: {}% ({}/{})'.format(
            (idx + 1) * 100 // num_samples, idx + 1, num_samples), end="\r")

    return numpy_specgrams


def convert_wav_to_mel_spec(
        path_to_wav,
        n_mels=constants.n_mels,
        fmax=constants.fmax,
        hop_length=constants.hop_length,
        sample_rate=constants.sample_rate,
        n_fft=constants.n_fft):
    """
    Converts a raw wav to a Tensor mel spectrogram
        Raw wave shape: (samples, 1)
        Tensor mel spectrogram shape: (1, t, n_mels)
    """

    audio, sample_rate = librosa.load(path_to_wav, sr=sample_rate)
    librosa_melspec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmax=fmax,
        center=True)

    return librosa_melspec_to_tf(librosa_melspec)


def convert_wav_to_stft_spec(
        path_to_wav,
        fmax=constants.fmax,
        win_length=constants.win_length,
        hop_length=constants.hop_length,
        sample_rate=constants.sample_rate,
        n_fft=constants.n_fft):

    audio, sample_rate = librosa.load(path_to_wav, sr=sample_rate)
    stft = librosa.core.stft(
        audio,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        center=False)

    return librosa_melspec_to_tf(stft)


def convert_mel_spec_to_wav(
        tf_melspec,
        fmax=constants.fmax,
        hop_length=constants.hop_length,
        sample_rate=constants.sample_rate,
        n_fft=constants.n_fft):
    """
    Converts a Tensor mel spectrogram to a Tensor wav
        Tensor mel spectrogram shape: (1, t, n_mels)
        Tensor wav shape: (1, t)
    """

    librosa_melspec = tf_melspec_to_librosa(tf_melspec)
    librosa_wav = librosa.feature.inverse.mel_to_audio(
        librosa_melspec.numpy(),
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        fmax=fmax,
        center=True
    )

    return librosa_wav_to_tf(librosa_wav)


def convert_stft_spec_to_wav(
        tf_melspec,
        win_length=constants.win_length,
        hop_length=constants.hop_length,
        sample_rate=constants.sample_rate):

    stft = tf_melspec_to_librosa(tf_melspec)
    audio = librosa.core.istft(
        stft.numpy(),
        hop_length=hop_length,
        win_length=win_length,
        center=False
    )

    return librosa_wav_to_tf(audio)


def tf_melspec_to_librosa(tf_melspec):
    """
    Converts a Tensorflow Tensor mel spectrogram to a Librosa mel spectrogram
        Tensor mel spectrogram shape: (1, t, n_mels)
        Librosa mel spectrogram shape: (n_mels, t)
    """
    return tf.transpose(tf_melspec[0])


def librosa_melspec_to_tf(librosa_melspec):
    """
    Converts a tensorflow Librosa mel spectrogram to a Tensorflow mel spectrogram
        Tensor mel spectrogram shape: (1, t, n_mels)
        Librosa mel spectrogram shape: (n_mels, t)
    """
    return tf.expand_dims(tf.transpose(tf.convert_to_tensor(librosa_melspec)), 0)


def librosa_wav_to_tf(librosa_wav, sample_rate=constants.sample_rate):
    """
    Converts a Librosa wav file to a Tensorflow Tensor
        Librosa wav shape: t
        Tensorflow wav shape: (1, t)
    """
    audio = tf.transpose(tf.expand_dims(tf.convert_to_tensor(librosa_wav), 0))
    return tf.audio.encode_wav(audio, sample_rate)


def tf_wav_to_librosa(tf_wav, sample_rate=constants.sample_rate):
    audio, sample_rate = tf.audio.decode_wav(tf_wav)
    return audio[0]
