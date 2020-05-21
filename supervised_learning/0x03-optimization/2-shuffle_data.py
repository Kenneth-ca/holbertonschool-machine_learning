#!/usr/bin/env python3
"""
Shuffles the data points in two matrices the same way
"""
import numpy as np


def shuffle_data(X, Y):
    """
    a function that shuffles the data points in two matrices the same way
    :param X: the first numpy.ndarray of shape (m, nx) to shuffle
    :param Y: the second numpy.ndarray of shape (m, ny) to shuffle
    :return: the shuffled X and Y matrices
    """
    permutation = np.random.permutation(X.shape[0])
    return X[permutation, :], Y[permutation, :]
