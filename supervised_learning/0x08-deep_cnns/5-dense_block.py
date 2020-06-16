#!/usr/bin/env python3
"""
Builds a dense block
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    a function that builds a dense block
    :param X: is the output from the previous layer
    :param nb_filters: is an integer representing the number of filters in X
    :param growth_rate: is the growth rate for the dense block
    :param: layers: is the number of layers in the dense block
    :return: The concatenated output of each layer within the Dense Block and
    the number of filters within the concatenated outputs, respectively
    """
    init = K.initializers.he_normal()

    for i in range(layers):
        norm0 = K.layers.BatchNormalization()(X)
        act0 = K.layers.Activation("relu")(norm0)
        bottle = K.layers.Conv2D(filters=4*growth_rate, kernel_size=(1, 1),
                                 padding="same",
                                 strides=(1, 1),
                                 kernel_initializer=init)(act0)

        norm1 = K.layers.BatchNormalization()(bottle)
        act1 = K.layers.Activation("relu")(norm1)
        conv = K.layers.Conv2D(filters=growth_rate, kernel_size=(3, 3),
                               padding="same",
                               strides=(1, 1),
                               kernel_initializer=init)(act1)

        X = K.layers.concatenate([X, conv])
        nb_filters += growth_rate

    return X, nb_filters
