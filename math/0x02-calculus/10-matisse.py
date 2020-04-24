#!/usr/bin/env python3
"""
A function that calculates the derivative of a polynomial
"""


def poly_derivative(poly):
    """
    Returns a list of the derivative
    >>> poly = [5, 3, 0, 1]
    >>> print(poly_derivative(poly))
    [3, 0, 3]
    """
    if type(poly) is not list or poly == []:
        return None
    elif len(poly) < 2:
        return [0]
    else:
        exponent = 1
        derivative = poly.copy()
        derivative.pop(0)
        for i in range(len(derivative)):
            if type(derivative[i]) is int or type(derivative[i]) is float:
                derivative[i] = derivative[i] * exponent
                exponent += 1
            else:
                return None
        return derivative
