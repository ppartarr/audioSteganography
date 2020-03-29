#!/usr/bin/env python3
# coding: utf-8

import tensorflow.keras.utils
import constants


class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, x_data, samples=constants.num_samples, batch_size=constants.batch_size):
        self.samples = samples
        self.x_data = x_data
        self.batch_size = batch_size
        self.secret = x_data[0]
        self.cover = x_data[1]

    def __len__(self):
        """
        Return the number of batches per epoch
        """
        return self.samples // self.batch_size

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        secret_batch = self.secret[
            index * self.batch_size:(index + 1) * self.batch_size]
        cover_batch = self.cover[
            index * self.batch_size:(index + 1) * self.batch_size]

        return ([secret_batch, cover_batch], [secret_batch, cover_batch])
