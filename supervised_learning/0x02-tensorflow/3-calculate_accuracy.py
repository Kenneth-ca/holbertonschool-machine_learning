#!/usr/bin/env python3
"""
Module to calculate accuracy of prediction
"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    a function that calculates the accuracy of a prediction
    :param y: a placeholders with the right labels of the input data
    :param y_pred: tensor containing the network's predictions
    :return: a tensor containing the decimal accuracy of the prediction
    """
    accuracy = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    mean = tf.reduce_mean(tf.cast(accuracy, tf.float32))
    return mean
