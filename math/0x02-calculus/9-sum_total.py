#!/usr/bin/env python3
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
    if (type(n) is not int) or (n is None):
        return None
    elif n < 1:
        return 0
    else:
        return n ** 2 + summation_i_squared(n - 1)
