#!/usr/bin/env python3
"""
Calculates the normalization
"""


def normalization_constants(X):
    """
    a function that calculates the normalization (standardization) constants
    of a matrix
    :param X: numpy.ndarray of shape (m, nx) to normalize
    :return: the mean and standard deviation of each feature, respectively
    """
    return X.mean(axis=0), X.std(axis=0)
