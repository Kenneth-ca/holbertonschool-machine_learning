#!/usr/bin/env python3
"""
Creates the class MultiNormal
"""
import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution
    """

    def __init__(self, data):
        """
        class constructor
        :param data: numpy.ndarray of shape (d, n) containing the data set:
            n is the number of data points
            d is the number of dimensions in each data point
        """
        if type(data) != np.ndarray:
            raise TypeError("data must be a 2D numpy.ndarray")
        if len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = data.mean(axis=1).reshape(d, 1)
        X_mean = data - self.mean
        self.cov = (X_mean @ X_mean.T) / (n - 1)
