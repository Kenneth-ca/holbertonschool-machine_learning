#!/usr/bin/env python3
"""
Converts a label vector into a one-hot matrix
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    a function that converts a label vector into a one-hot matrix
    :param labels: label vector
    :param classes: classes for the one-hot matrix
    :return: the one-hot matrix
    """
    oh_encode = K.utils.to_categorical(labels, num_classes=classes)
    return oh_encode
