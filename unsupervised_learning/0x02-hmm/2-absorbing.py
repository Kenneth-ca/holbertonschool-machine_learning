#!/usr/bin/env python3
"""
Determines if a markov chain is absorbing
"""
import numpy as np


def absorbing(P):
    """
    Determines if a markov chain is absorbing
    :param P: square 2D numpy.ndarray of shape (n, n) representing the
    transition matrix
    :return: True if it is absorbing, or False on failure
    """
    if type(P) is not np.ndarray:
        return False
    if len(P.shape) != 2:
        return False
    n, n_t = P.shape
    if n != n_t:
        return False
    sum_test = np.sum(P, axis=1)
    for elem in sum_test:
        if not np.isclose(elem, 1):
            return False

    diagonal = np.diag(P)
    if (diagonal == 1).all():
        return True

    absorb = (diagonal == 1)
    for row in range(len(diagonal)):
        for col in range(len(diagonal)):
            if P[row, col] > 0 and absorb[col]:
                absorb[row] = 1
    if (absorb == 1).all():
        return True

    return False
