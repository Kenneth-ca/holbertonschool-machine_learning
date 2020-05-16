#!/usr/bin/env python3
"""
Module to calculate the loss of a prediction
"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    a function that calculates the loss of a prediction
    :param y: a placeholders with the right labels of the input data
    :param y_pred: tensor containing the network's predictions
    :return: a tensor containing the loss of a prediction
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)
