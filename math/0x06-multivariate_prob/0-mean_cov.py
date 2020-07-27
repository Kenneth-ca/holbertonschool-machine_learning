#!/usr/bin/env python3
"""
Calculates the mean and covariance of a data set
"""
import numpy as np


def mean_cov(X):
    """
    a function that calculates the mean and covariance
    :param X: numpy.ndarray of shape (n, d) containing the data set:
        n is the number of data points
        d is the number of dimensions in each data point
        If X is not a 2D numpy.ndarray, raise a TypeError with the message X
        must be a 2D numpy.ndarray
        If n is less than 2, raise a ValueError with the message X must contain
        multiple data points
    :return: mean, cov:
        mean is a numpy.ndarray of shape (1, d) containing the mean of the data
        set
        cov is a numpy.ndarray of shape (d, d) containing the covariance matrix
        of the data set
    """
    if type(X) is not np.ndarray:
        raise TypeError("X must be a 2D numpy.ndarray")
    if len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = X.mean(axis=0).reshape(1, d)
    X_mean = X - mean
    cov = (X_mean.T @ X_mean) / (n - 1)
    return mean, cov
