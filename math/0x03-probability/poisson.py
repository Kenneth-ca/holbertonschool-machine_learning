#!/usr/bin/env python3
"""
Representing poisson distribution
"""


class Poisson:
    """
    Class poisson distribution
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
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """
        Probability Mass Function for poisson
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        e_mean = Poisson.e ** - self.lambtha
        mean_k = self.lambtha ** k
        x_factorial = 1
        for i in range(1, k + 1):
            x_factorial *= i
        pmf = e_mean * mean_k / x_factorial
        return pmf

    def cdf(self, k):
        """
        Cumulative Distribution Function for poisson
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += Poisson.pmf(self, i)
        return cdf
