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
    try:
        encode = np.zeros((classes, Y.shape[0]))
        encode[Y, np.arange(Y.shape[0])] = 1
        return encode
    except Exception:
        return None
