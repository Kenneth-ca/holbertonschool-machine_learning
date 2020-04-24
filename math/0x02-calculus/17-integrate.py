#!/usr/bin/env python3
"""
A function that calculates the integral of a polynomial
"""


def poly_integral(poly, C=0):
    """
    Returns a list of the integral
    >>> poly = [5, 3, 0, 1]
    >>> print(poly_integral(poly))
    [0, 5, 1.5, 0, 0.25]
    """
    if type(poly) is not list:
        return None
    elif type(C) is int or type(C) is float:
        exponent = 0
        derivative = poly.copy()
        for i in range(len(derivative)):
            if type(derivative[i]) is int or type(derivative[i]) is float:
                exponent += 1
                number = derivative[i] / exponent
                derivative[i] = int(number) if number % 1 == 0 else number
            else:
                return None
        derivative.insert(0, C)
        return derivative
    else:
        return None
