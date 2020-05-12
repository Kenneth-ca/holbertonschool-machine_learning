#!/usr/bin/env python3
"""
Converts a one-hot matrix into a numeric label vector:
"""
import numpy as np


def one_hot_decode(one_hot):
    """
    converts to a vector of labels
    :param one_hot: a one-hot encoded numpy.ndarray with shape (classes, m)
    :return: a numpy.ndarray with shape (m, ) containing the numeric labels
    for each example, or None on failure
    """
    if type(one_hot) is not np.ndarray:
        return None
    if len(one_hot) == 0 or len(one_hot.shape) != 2:
        return None
    labels = np.argmax(one_hot, axis=0)
    return labels
