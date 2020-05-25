#!/usr/bin/env python3
"""
Uses batch normalization with tensorflow
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    a function that uses batch normalization with tensorflow
    :param prev: the activated output of the previous layer
    :param n: the number of nodes in the layer to be created
    :param activation: the activation function that should be used on the
    output of the layer
    :return: a tensor of the activated output for the layer
    """
    # Layers
    k_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    output = tf.layers.Dense(units=n, kernel_initializer=k_init)
    Z = output(prev)

    # Gamma and Beta initialization
    gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[n]),
                        name="gamma")
    beta = tf.Variable(initial_value=tf.constant(0.0, shape=[n]), name="beta")

    # Batch normalization
    mean, var = tf.nn.moments(Z, axes=0)
    b_norm = tf.nn.batch_normalization(Z, mean, var, offset=beta,
                                       scale=gamma,
                                       variance_epsilon=1e-8)
    if activation is None:
        return b_norm
    else:
        return activation(b_norm)
