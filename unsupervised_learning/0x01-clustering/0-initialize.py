#!/usr/bin/env python3
"""
Initializes cluster centroids for K-means
"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means
    :param X: numpy.ndarray of shape (n, d) containing the dataset that will
    be used for K-means clustering
        n is the number of data points
        d is the number of dimensions for each data point
    :param k: positive integer containing the number of clusters
    :return: numpy.ndarray of shape (k, d) containing the initialized
    centroids for each cluster, or None on failure
    """
    if type(k) is not int or k <= 0:
        return None
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    n, d = X.shape
    clusters = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0),
                                 size=(k, d))
    return clusters
