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
    if (type(n) is not int) or (n is None) or (n < 1):
        return None
    else:
        numbers = range(1, n + 1)
        result = 0
        result = map(lambda i: i ** 2, numbers)
        return sum(result)
