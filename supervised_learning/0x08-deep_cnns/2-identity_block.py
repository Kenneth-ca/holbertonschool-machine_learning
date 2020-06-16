#!/usr/bin/env python3
"""
Builds an identity block
"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    a function that builds an identity block
    :param A_prev: is the output from the previous layer
    :param filters: is a tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution
    :return: the concatenated output of the identity block
    """
    init = K.initializers.he_normal()
    F11, F3, F12 = filters

    # First layer
    c_F11 = K.layers.Conv2D(F11, kernel_size=1, padding="same",
                            kernel_initializer=init)(A_prev)
    norm_F11 = K.layers.BatchNormalization()(c_F11)
    act_F11 = K.layers.Activation("relu")(norm_F11)

    # Second layer
    c_F3 = K.layers.Conv2D(F3, kernel_size=3, padding="same",
                           kernel_initializer=init)(act_F11)
    norm_F3 = K.layers.BatchNormalization()(c_F3)
    act_F3 = K.layers.Activation("relu")(norm_F3)

    # Third layer
    c_F12 = K.layers.Conv2D(F12, kernel_size=1, padding="same",
                            kernel_initializer=init)(act_F3)
    norm_F12 = K.layers.BatchNormalization()(c_F12)

    # Add shortcut
    X = K.layers.Add()([norm_F12, A_prev])

    act = K.layers.Activation("relu")(X)

    return act
