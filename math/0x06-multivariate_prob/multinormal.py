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

    def pdf(self, x):
        """
        calculates the PDF
        :param x: numpy.ndarray of shape (d, 1) containing the data point whose
        PDF should be calculated
            d is the number of dimensions of the Multinomial instance
        :return: the value of the PDF
        """
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2 or x.shape[1] != 1 or x.shape[0] != d:
            raise ValueError("x must have the shape ({}, 1)".format(d))

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        first = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(det))
        second = np.dot((x - self.mean).T, inv)
        third = np.dot(second, (x - self.mean) / -2)
        pdf = first * np.exp(third)

        return pdf[0][0]
