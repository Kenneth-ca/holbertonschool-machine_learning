#!/usr/bin/env python3
"""
Representing exponential distribution
"""


class Exponential:
    """
    Class exponential distribution
    """
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        Class contructor
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1 / (sum(data) / len(data))

    def pdf(self, x):
        """
        Probability Density Function for exponential
        """
        if x < 0:
            return 0
        pdf = self.lambtha * Exponential.e ** - (self.lambtha * x)
        return pdf

    def cdf(self, x):
        """
        Cumulative Distribution Function for exponential
        """
        if x < 0:
            return 0
        cdf = 1 - Exponential.e ** (- self.lambtha * x)
        return cdf
