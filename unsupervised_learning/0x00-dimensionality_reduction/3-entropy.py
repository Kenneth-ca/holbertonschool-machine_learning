#!/usr/bin/env python3
"""
Calculates the entropy
"""
import numpy as np


def HP(Di, beta):
    """
    Calculates the Shannon entropy and P affinities relative to a data point
    :param Di: numpy.ndarray of shape (n - 1,) containing the pariwise
    distances between a data point and all other points except itself
        n is the number of data points
    :param beta: the beta value for the Gaussian distribution
    :return: (Hi, Pi)
        Hi: the Shannon entropy of the points
        Pi: a numpy.ndarray of shape (n - 1,) containing the P affinities of
        the points
    """
    num = (np.exp(- Di.copy() * beta))
    den = (np.sum(np.exp(-Di.copy() * beta)))
    P = num / den

    Hi = - np.sum(P * np.log2(P))

    return Hi, P
