#!/usr/bin/env python3
"""
Converts a numeric label vector into a one-hot matrix:
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    converts to one-hot-matrix
    :param Y: numpy.ndarray with shape (m,) containing numeric class labels
    :param classes: the maximum number of classes found in Y
    :return: a one-hot encoding of Y with shape (classes, m), or None on
    failure
    """
    if type(classes) is not int or classes <= Y.max():
        return None
    if type(Y) is not np.ndarray or len(Y) < 1:
        return None
    if Y.size == 0 or Y.min() < 0:
        return None
    one_hot = np.zeros((classes, Y.shape[0]))
    one_hot[Y, np.arange(Y.shape[0])] = 1
    return one_hot
