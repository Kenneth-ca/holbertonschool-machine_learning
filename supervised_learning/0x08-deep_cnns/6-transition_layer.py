#!/usr/bin/env python3
"""
Builds a transition layer
"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    a function that builds a transition layer
    :param X: is the output from the previous layer
    :param nb_filters: is an integer representing the number of filters in X
    :param compression: is the compression factor for the transition layer
    :return: The concatenated output of each layer within the Dense Block and
    the number of filters within the concatenated outputs, respectively
    """
    init = K.initializers.he_normal()
    n_number = int(nb_filters * compression)

    norm0 = K.layers.BatchNormalization()(X)
    act0 = K.layers.Activation("relu")(norm0)
    conv = K.layers.Conv2D(filters=int(n_number),
                           kernel_size=(1, 1),
                           padding="same",
                           strides=(1, 1),
                           kernel_initializer=init)(act0)

    avg_pool = K.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2),
                                         padding="same")(conv)

    return avg_pool, n_number
