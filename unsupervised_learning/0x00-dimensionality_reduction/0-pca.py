#!/usr/bin/env python3
"""
Performs PCA on a dataset
"""
import numpy as np


def pca(X, var=0.95):
    """
    a function that performs PCA on a dataset
    :param X: numpy.ndarray of shape (n, d) where:
        n is the number of data points
        d is the number of dimensions in each point
    :param var: the fraction of the variance that the PCA transformation
    should maintain
    :return: the weights matrix, W, that maintains var fraction of X‘s
    original variance
    """
    u, s, vh = np.linalg.svd(X)
    cum = np.cumsum(s)
    thresh = cum[len(cum) - 1] * var
    mask = np.where(thresh > cum)
    var = cum[mask]
    idx = len(var) + 1
    W = vh.T
    Wr = W[:, 0:idx]
    return Wr
