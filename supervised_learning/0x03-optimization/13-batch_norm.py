#!/usr/bin/env python3
"""
Normalizes an unactivated output of a neural network using batch normalization
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    a function that uses batch normalization
    :param Z: numpy.ndarray of shape (m, n) that should be normalized
        m is the number of data points
        n is the number of features in Z
    :param gamma: numpy.ndarray of shape (1, n) containing the scales used
    for batch normalization
    :param beta: numpy.ndarray of shape (1, n) containing the offsets used
    for batch normalization
    :param epsilon: a small number to avoid division by zero
    :return: the normalized Z matrix
    """
    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    normalized = (Z - mean) / np.sqrt(var + epsilon)
    z_n = gamma * normalized + beta
    return z_n
