#!/usr/bin/env python3
"""
Normalizes (standardizes) a matrix
"""


def normalize(X, m, s):
    """
    a function that normalizes (standardizes) a matrix
    :param X: numpy.ndarray of shape (d, nx) to normalize
    where d is the number of data point, nx is the number of features
    :param m: numpy.ndarray of shape (nx,) that contains the mean of all
    features of X
    :param s: numpy.ndarray of shape (nx,) that contains the standard
    deviation of all features of X
    :return: The normalized X matrix
    """
    return (X - m) / s
