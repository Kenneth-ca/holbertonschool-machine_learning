#!/usr/bin/env python3
"""
Performs K-means
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset
    :param X: numpy.ndarray of shape (n, d) containing the dataset that will
    be used for K-means clustering
        n is the number of data points
        d is the number of dimensions for each data point
    :param iterations: positive integer containing the maximum number of
    iterations that should be performed
    :return: C, clss, or None, None on failure
        C is a numpy.ndarray of shape (k, d) containing the centroid means
        for each cluster
        clss is a numpy.ndarray of shape (n,) containing the index of the
        cluster in C that each data point belongs to
    """
    if type(k) is not int or k <= 0:
        return None, None
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    n, d = X.shape
    centroids = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0),
                                  size=(k, d))
    for i in range(iterations):
        copy = centroids.copy()
        D = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        clss = np.argmin(D, axis=0)
        for j in range(k):
            if len(X[clss == j]) == 0:
                centroids[j] = np.random.uniform(np.min(X, axis=0),
                                                 np.max(X, axis=0),
                                                 size=(1, d))
            else:
                centroids[j] = (X[clss == j]).mean(axis=0)
        D = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        clss = np.argmin(D, axis=0)
        if np.all(copy == centroids):
            return centroids, clss

    return centroids, clss
