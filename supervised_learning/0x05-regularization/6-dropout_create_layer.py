#!/usr/bin/env python3
"""
Creates a layer of a neural network using dropout
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    a function that creates a layer of a NN using dropout
    :param prev: tensor containing the output of the previous layer
    :param n: the number of nodes the new layer should contain
    :param activation: the activation function that should be used on the layer
    :param keep_prob: the probability that a node will be kept
    :return: the output of the new layer
    """
    dropout = tf.layers.Dropout(keep_prob)
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, activation=activation, kernel_initializer=init,
                            kernel_regularizer=dropout)
    return layer(prev)
