#!/usr/bin/env python3
"""
Calculates the intersection
"""
import numpy as np


def marginal(x, n, P, Pr):
    """
    Calculates the intersection of obtaining this data with the various
    hypothetical probabilities
    :param x: number of patients that develop severe side effects
    :param n: total number of patients observed
    :param P: 1D numpy.ndarray containing the various hypothetical
    probabilities of developing severe side effects
    :param Pr: 1D numpy.ndarray containing the prior beliefs of P
    :return: numpy.ndarray containing the likelihood of obtaining the data,
    x and n, for each probability in P, respectively
    """
    if (type(n) is not int) or (n <= 0):
        raise ValueError("n must be a positive integer")
    if (type(x) is not int) or (x < 0):
        raise ValueError("x must be an integer that is greater than or equal "
                         "to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")

    if (type(P) is not np.ndarray) or (len(P.shape) != 1):
        raise TypeError("P must be a 1D numpy.ndarray")
    if (type(Pr) is not np.ndarray) or (P.shape != Pr.shape):
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    for p, pr in zip(P, Pr):
        if not (p >= 0 and p <= 1):
            raise ValueError("All values in P must be in the range [0, "
                             "1]")
        if not (pr >= 0 and pr <= 1):
            raise ValueError("All values in Pr must be in the range [0, "
                             "1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    num = np.math.factorial(n)
    den = np.math.factorial(x) * np.math.factorial(n - x)

    like = num / den * (P ** x) * ((1 - P) ** (n - x))
    inter = like * Pr
    marg = np.sum(inter)

    return marg
