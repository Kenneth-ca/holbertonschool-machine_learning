#!/usr/bin/python3
"""
A function that calculates a cuadratic sum
"""


def summation_i_squared(n):
    """
    Returns the result of the sum for some n
    >>> n = 5
    >>> print(summation_i_squared(n))
    55
    """
    return 0 if n < 1 else n ** 2 + summation_i_squared(n - 1)
