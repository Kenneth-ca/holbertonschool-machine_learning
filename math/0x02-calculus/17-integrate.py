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
    if type(poly) is not list or len(poly) == 0:
        return None
    elif type(C) is int:
        if poly == [0]:
            return [C]
        exponent = 0
        integral = poly.copy()
        for i in range(len(integral)):
            if type(integral[i]) is int or type(integral[i]) is float:
                exponent += 1
                number = integral[i] / exponent
                integral[i] = int(number) if number % 1 == 0 else number
            else:
                return None
        integral.insert(0, C)
        return integral
    else:
        return None
