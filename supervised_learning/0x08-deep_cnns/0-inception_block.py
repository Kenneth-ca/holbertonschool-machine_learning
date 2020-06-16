#!/usr/bin/env python3
"""
Builds an inception block
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    a function that builds an inception block
    :param A_prev: is the output from the previous layer
    :param filters: is a tuple or list containing F1, F3R, F3,F5R, F5, FPP,
    respectively:
        F1 is the number of filters in the 1x1 convolution
        F3R is the number of filters in the 1x1 convolution before the 3x3
        convolution
        F3 is the number of filters in the 3x3 convolution
        F5R is the number of filters in the 1x1 convolution before the 5x5
        convolution
        F5 is the number of filters in the 5x5 convolution
        FPP is the number of filters in the 1x1 convolution after the max
        pooling
    :return: the concatenated output of the inception block
    """
    init = K.initializers.he_normal()
    F1, F3R, F3, F5R, F5, FPP = filters

    c_F1 = K.layers.Conv2D(F1, kernel_size=1, padding="same",
                           activation="relu", kernel_initializer=init)(A_prev)

    c_F3R = K.layers.Conv2D(F3R, kernel_size=1, padding="same",
                            activation="relu", kernel_initializer=init)(A_prev)

    c_F3 = K.layers.Conv2D(F3, kernel_size=3, padding="same",
                           activation="relu", kernel_initializer=init)(c_F3R)

    c_F5R = K.layers.Conv2D(F5R, kernel_size=1, padding="same",
                            activation="relu", kernel_initializer=init)(A_prev)

    c_F5 = K.layers.Conv2D(F5, kernel_size=5, padding="same",
                           activation="relu", kernel_initializer=init)(c_F5R)

    pool = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                                 padding="same")(A_prev)

    c_FPP = K.layers.Conv2D(FPP, kernel_size=1, padding="same",
                            activation="relu", kernel_initializer=init)(pool)

    concat = K.layers.concatenate([c_F1, c_F3, c_F5, c_FPP])

    return concat
