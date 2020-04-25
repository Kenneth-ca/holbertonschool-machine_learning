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
        result = 0
        for i in range(n + 1):
            result = (i ** 2) + result
        return result
