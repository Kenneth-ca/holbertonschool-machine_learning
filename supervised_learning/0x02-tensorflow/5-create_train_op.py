#!/usr/bin/env python3
"""
Module to create the training operation
"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """
    a function creates the training operation for the network:
    :param loss: loss of the network's prediction
    :param alpha: the learning rate
    :return: an operation that trains the network using gradient descent
    """
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
