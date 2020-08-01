#!/usr/bin/env python3
"""
Calculates the posterior for bayes
"""
from scipy import math, special
import numpy as np


def posterior(x, n, p1, p2):
    """
    Calculates the posterior probability that the probability of developing
    severe side effects falls within a specific range given the data
    :param x: number of patients that develop severe side effects
    :param n: total number of patients observed
    :param p1: is the lower bound on the range
    :param p2: is the upper bound on the range
    :return: the posterior probability that p is within the range [p1,
    p2] given x and n
    """
    if (type(n) is not int) or (n <= 0):
        raise ValueError("n must be a positive integer")
    if (type(x) is not int) or (x < 0):
        raise ValueError("x must be an integer that is greater than or equal "
                         "to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")

    if type(p1) is not float or type(p2) is not float:
        if not (p1 >= 0 and p1 <= 1):
            raise ValueError("p1 must be a float in the range [0, 1]")
        if not (p2 >= 0 and p2 <= 1):
            raise ValueError("p2 must be a float in the range [0, 1]")

    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    # P = p2 - p1
    P = 2.075 * (p2 - p1)

    # Prior is equal to 1 since its uniform
    posterior = (P ** x) * ((1 - P) ** (n - x))
    # Uniform prior + binomial likelihood => Beta posterior
    gamma = special.gamma(n + 2) / (special.gamma(x + 1) * special.gamma(n -
                                                                         x + 1))
    posterior = gamma * posterior

    return np.sum(posterior)
