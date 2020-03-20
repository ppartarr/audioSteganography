#!/usr/bin/env python3
# coding: utf-8

from tensorflow.keras.layers import Input, Conv2D, concatenate, GaussianNoise
from tensorflow.keras.models import Model
from tensorflow.keras import losses

import constants


def steg_model(input_shape, pretrain=False):

    lossFns = {
        "hide_conv_f": losses.mean_squared_error,
        "revl_conv_f": losses.mean_squared_error,
    }
    lossWeights = {
        "hide_conv_f": 1.0,
        "revl_conv_f": 0.8
    }

    # Inputs
    secret = Input(shape=input_shape, name='secret')
    cover = Input(shape=input_shape, name='cover')

    # Prepare network - patches [3*3,3*3,3*3]
    pconv_3x3 = Conv2D(64, kernel_size=3, padding="same",
                       activation='relu', name='prep_conv3x3_1')(secret)
    pconv_3x3 = Conv2D(64, kernel_size=3, padding="same",
                       activation='relu', name='prep_conv3x3_2')(pconv_3x3)
    pconv_3x3 = Conv2D(64, kernel_size=3, padding="same",
                       activation='relu', name='prep_conv3x3_3')(pconv_3x3)
    pconv_3x3 = Conv2D(64, kernel_size=3, padding="same",
                       activation='relu', name='prep_conv3x3_4')(pconv_3x3)

    pconv_4x4 = Conv2D(64, kernel_size=4, padding="same",
                       activation='relu', name='prep_conv4x4_1')(secret)
    pconv_4x4 = Conv2D(64, kernel_size=4, padding="same",
                       activation='relu', name='prep_conv4x4_2')(pconv_4x4)
    pconv_4x4 = Conv2D(64, kernel_size=4, padding="same",
                       activation='relu', name='prep_conv4x4_3')(pconv_4x4)
    pconv_4x4 = Conv2D(64, kernel_size=4, padding="same",
                       activation='relu', name='prep_conv4x4_4')(pconv_4x4)

    pconv_5x5 = Conv2D(64, kernel_size=5, padding="same",
                       activation='relu', name='prep_conv5x5_1')(secret)
    pconv_5x5 = Conv2D(64, kernel_size=5, padding="same",
                       activation='relu', name='prep_conv5x5_2')(pconv_5x5)
    pconv_5x5 = Conv2D(64, kernel_size=5, padding="same",
                       activation='relu', name='prep_conv5x5_3')(pconv_5x5)
    pconv_5x5 = Conv2D(64, kernel_size=5, padding="same",
                       activation='relu', name='prep_conv5x5_4')(pconv_5x5)

    pconcat_1 = concatenate(
        [pconv_3x3, pconv_4x4, pconv_5x5], axis=3, name="prep_concat_1")

    pconv_5x5 = Conv2D(64, kernel_size=5, padding="same",
                       activation='relu', name='prep_conv5x5_f')(pconcat_1)
    pconv_4x4 = Conv2D(64, kernel_size=4, padding="same",
                       activation='relu', name='prep_conv4x4_f')(pconcat_1)
    pconv_3x3 = Conv2D(64, kernel_size=3, padding="same",
                       activation='relu', name='prep_conv3x3_f')(pconcat_1)

    pconcat_f1 = concatenate(
        [pconv_5x5, pconv_4x4, pconv_3x3], axis=3, name="prep_concat_2")

    # Hiding network - patches [3*3,3*3,3*3]
    hconcat_h = concatenate([cover, pconcat_f1], axis=3, name="hide_concat_1")

    hconv_3x3 = Conv2D(64, kernel_size=3, padding="same",
                       activation='relu', name='hide_conv3x3_1')(hconcat_h)
    hconv_3x3 = Conv2D(64, kernel_size=3, padding="same",
                       activation='relu', name='hide_conv3x3_2')(hconv_3x3)
    hconv_3x3 = Conv2D(64, kernel_size=3, padding="same",
                       activation='relu', name='hide_conv3x3_3')(hconv_3x3)
    hconv_3x3 = Conv2D(64, kernel_size=3, padding="same",
                       activation='relu', name='hide_conv3x3_4')(hconv_3x3)

    hconv_4x4 = Conv2D(64, kernel_size=4, padding="same",
                       activation='relu', name='hide_conv4x4_1')(hconcat_h)
    hconv_4x4 = Conv2D(64, kernel_size=4, padding="same",
                       activation='relu', name='hide_conv4x4_2')(hconv_4x4)
    hconv_4x4 = Conv2D(64, kernel_size=4, padding="same",
                       activation='relu', name='hide_conv4x4_3')(hconv_4x4)
    hconv_4x4 = Conv2D(64, kernel_size=4, padding="same",
                       activation='relu', name='hide_conv4x4_4')(hconv_4x4)

    hconv_5x5 = Conv2D(64, kernel_size=5, padding="same",
                       activation='relu', name='hide_conv5x5_1')(hconcat_h)
    hconv_5x5 = Conv2D(64, kernel_size=5, padding="same",
                       activation='relu', name='hide_conv5x5_2')(hconv_5x5)
    hconv_5x5 = Conv2D(64, kernel_size=5, padding="same",
                       activation='relu', name='hide_conv5x5_3')(hconv_5x5)
    hconv_5x5 = Conv2D(64, kernel_size=5, padding="same",
                       activation='relu', name='hide_conv5x5_4')(hconv_5x5)

    hconcat_1 = concatenate(
        [hconv_3x3, hconv_4x4, hconv_5x5], axis=3, name="hide_concat_2")

    hconv_5x5 = Conv2D(64, kernel_size=5, padding="same",
                       activation='relu', name='hide_conv5x5_f')(hconcat_1)
    hconv_4x4 = Conv2D(64, kernel_size=4, padding="same",
                       activation='relu', name='hide_conv4x4_f')(hconcat_1)
    hconv_3x3 = Conv2D(64, kernel_size=3, padding="same",
                       activation='relu', name='hide_conv3x3_f')(hconcat_1)

    hconcat_f1 = concatenate(
        [hconv_5x5, hconv_4x4, hconv_3x3], axis=3, name="hide_concat_3")

    cover_pred = Conv2D(constants.num_mel_filters, kernel_size=1, padding="same",
                        name='hide_conv_f')(hconcat_f1)

    # Noise layer
    noise_ip = GaussianNoise(0.1)(cover_pred)

    # Reveal network - patches [3*3,3*3,3*3]
    rconv_3x3 = Conv2D(64, kernel_size=3, padding="same",
                       activation='relu', name='revl_conv3x3_1')(noise_ip)
    rconv_3x3 = Conv2D(64, kernel_size=3, padding="same",
                       activation='relu', name='revl_conv3x3_2')(rconv_3x3)
    rconv_3x3 = Conv2D(64, kernel_size=3, padding="same",
                       activation='relu', name='revl_conv3x3_3')(rconv_3x3)
    rconv_3x3 = Conv2D(64, kernel_size=3, padding="same",
                       activation='relu', name='revl_conv3x3_4')(rconv_3x3)

    rconv_4x4 = Conv2D(64, kernel_size=4, padding="same",
                       activation='relu', name='revl_conv4x4_1')(noise_ip)
    rconv_4x4 = Conv2D(64, kernel_size=4, padding="same",
                       activation='relu', name='revl_conv4x4_2')(rconv_4x4)
    rconv_4x4 = Conv2D(64, kernel_size=4, padding="same",
                       activation='relu', name='revl_conv4x4_3')(rconv_4x4)
    rconv_4x4 = Conv2D(64, kernel_size=4, padding="same",
                       activation='relu', name='revl_conv4x4_4')(rconv_4x4)

    rconv_5x5 = Conv2D(64, kernel_size=5, padding="same",
                       activation='relu', name='revl_conv5x5_1')(noise_ip)
    rconv_5x5 = Conv2D(64, kernel_size=5, padding="same",
                       activation='relu', name='revl_conv5x5_2')(rconv_5x5)
    rconv_5x5 = Conv2D(64, kernel_size=5, padding="same",
                       activation='relu', name='revl_conv5x5_3')(rconv_5x5)
    rconv_5x5 = Conv2D(64, kernel_size=5, padding="same",
                       activation='relu', name='revl_conv5x5_4')(rconv_5x5)

    rconcat_1 = concatenate(
        [rconv_3x3, rconv_4x4, rconv_5x5], axis=3, name="revl_concat_1")

    rconv_5x5 = Conv2D(64, kernel_size=5, padding="same",
                       activation='relu', name='revl_conv5x5_f')(rconcat_1)
    rconv_4x4 = Conv2D(64, kernel_size=4, padding="same",
                       activation='relu', name='revl_conv4x4_f')(rconcat_1)
    rconv_3x3 = Conv2D(64, kernel_size=3, padding="same",
                       activation='relu', name='revl_conv3x3_f')(rconcat_1)

    rconcat_f1 = concatenate(
        [rconv_5x5, rconv_4x4, rconv_3x3], axis=3, name="revl_concat_2")

    secret_pred = Conv2D(constants.num_mel_filters, kernel_size=1, padding="same",
                         name='revl_conv_f')(rconcat_f1)

    model = Model(inputs=[secret, cover], outputs=[cover_pred, secret_pred])

    # Compile model
    model.compile(optimizer='adam', loss=lossFns, loss_weights=lossWeights)

    return model
