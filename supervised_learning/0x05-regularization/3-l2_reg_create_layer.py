#!/usr/bin/env python3
"""
Creates a tensorflow layer with L2 regularization
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    a function that creates a tensrflow layer with L2 regularization
    :param prev: is a tensor containing the output of the previous layer
    :param n: the number of nodes the new layer should contain
    :param activation: the activation function that should be used on the layer
    :param lambtha: the L2 regularization parameter
    :return: the output of the new layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(n, activation=activation, kernel_initializer=init,
                            name="layer", kernel_regularizer=reg)
    return layer(prev)
