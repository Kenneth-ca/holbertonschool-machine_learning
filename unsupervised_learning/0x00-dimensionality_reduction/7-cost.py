#!/usr/bin/env python3
"""
Calculates the cost
"""
import numpy as np


def cost(P, Q):
    """
    Calculates the cost of the t-SNE transformation
    :param P: numpy.ndarray of shape (n, n) containing the P affinities
    :param Q: numpy.ndarray of shape (n, n) containing the Q affinities
    :return: C, the cost of the transformation
    """
    Q = np.maximum(Q, 1e-12)
    PQ = P / Q
    PQ = np.maximum(PQ, 1e-12)
    log = np.log(PQ)
    cost = np.sum(P * log)

    return cost
