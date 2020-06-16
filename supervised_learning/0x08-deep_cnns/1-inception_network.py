#!/usr/bin/env python3
"""
Builds an inception network
"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    a function that builds an inception network
    :return: the keras model
    """
    init = K.initializers.he_normal()
    X = K.Input(shape=(224, 224, 3))
    conv0 = K.layers.Conv2D(64, kernel_size=7, strides=(2, 2), padding="same",
                            kernel_initializer=init, activation="relu")(X)

    max_pool0 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                      padding="same")(conv0)

    conv1R = K.layers.Conv2D(64, kernel_size=1, strides=(1, 1), padding="same",
                             kernel_initializer=init, activation="relu")
    conv1R = conv1R(max_pool0)

    conv1 = K.layers.Conv2D(192, kernel_size=3, strides=(1, 1), padding="same",
                            kernel_initializer=init, activation="relu")
    conv1 = conv1(conv1R)

    max_pool1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                      padding="same")(conv1)

    incep0 = inception_block(max_pool1, [64, 96, 128, 16, 32, 32])

    incep1 = inception_block(incep0, [128, 128, 192, 32, 96, 64])

    max_pool2 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                      padding="same")(incep1)

    incep2 = inception_block(max_pool2, [192, 96, 208, 16, 48, 64])

    incep3 = inception_block(incep2, [160, 112, 224, 24, 64, 64])

    incep4 = inception_block(incep3, [128, 128, 256, 24, 64, 64])

    incep5 = inception_block(incep4, [112, 144, 288, 32, 64, 64])

    incep6 = inception_block(incep5, [256, 160, 320, 32, 128, 128])

    max_pool2 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                      padding="same")(incep6)

    incep7 = inception_block(max_pool2, [256, 160, 320, 32, 128, 128])

    incep8 = inception_block(incep7, [384, 192, 384, 48, 128, 128])

    avg_pool0 = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1),
                                          padding="valid")(incep8)

    dropout = K.layers.Dropout(rate=0.4)(avg_pool0)

    softmax = K.layers.Dense(units=1000, activation="softmax",
                             kernel_initializer=init)(dropout)

    model = K.Model(inputs=X, outputs=softmax)

    return model
