#!/usr/bin/env python3
"""
Task statement: You are conducting a study on a revolutionary cancer drug and
are looking to find the probability that a patient who takes this drug will
develop severe side effects. During your trials, n patients take the drug and
x patients develop severe side effects. You can assume that x follows a
binomial distribution.
"""
import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining this data given various
    hypothetical probabilities of developing severe side effects
    :param x: number of patients that develop severe side effects
    :param n: total number of patients observed
    :param P: 1D numpy.ndarray containing the various hypothetical
    probabilities of developing severe side effects
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
    for p in P:
        if not (p >= 0 and p <= 1):
            raise ValueError("All values in P must be in the range [0, 1]")

    num = np.math.factorial(n)
    den = np.math.factorial(x) * np.math.factorial(n - x)

    like = num / den * (P ** x) * ((1 - P) ** (n - x))

    return like
