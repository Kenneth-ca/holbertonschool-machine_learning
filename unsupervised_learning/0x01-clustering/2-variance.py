#!/usr/bin/env python3
"""
Calculates the total intra-cluster variance for a data set
"""
import numpy as np


def variance(X, C):
    """
    Calculates the intra-cluster variance
    :param X: numpy.ndarray of shape (n, d) containing the data set
    :param C: numpy.ndarray of shape (k, d) containing the centroid means for
    each cluster
    :return: var, or None on failure
        var is the total variance
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(C) is not np.ndarray or len(X.shape) != 2:
        return None, None
    D = np.sqrt(((X - C[:, np.newaxis]) ** 2).sum(axis=2))
    cluster = np.min(D, axis=0)

    var = np.sum(np.square(cluster))
    return var
