#!/usr/bin/env python3
"""
Calculates the correlation matrix
"""
import numpy as np


def correlation(C):
    """
    a function that calculates the correlation matrix
    :param C: numpy.ndarray of shape (d, d) containing a covariance matrix
        d is the number of dimensions
        If C is not a numpy.ndarray, raise a TypeError with the message C must
        be a numpy.ndarray
        If C does not have shape (d, d), raise a ValueError with the message C
        must bea 2D square matrix
    :return: numpy.ndarray of shape (d, d) containing the correlation matrix
    """
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    diag = np.diag(C).reshape(1, -1)
    stddev = np.sqrt(diag)

    corr = C / (stddev * stddev.T)
    return corr
