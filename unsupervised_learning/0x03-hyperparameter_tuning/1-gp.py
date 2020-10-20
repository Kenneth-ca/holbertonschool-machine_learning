#!/usr/bin/env python3
"""
Create the class GaussianProcess that represents a noiseless 1D
Gaussian process
"""
import numpy as np


class GaussianProcess:
    """
    A class that represents a Gaussian process
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        A function that initializes a class
        :param X_init: numpy.ndarray of shape (t, 1) representing the inputs
        already sampled with the black-box function
        :param Y_init: numpy.ndarray of shape (t, 1) representing the outputs
        of the black-box function for each input in X_init
        :param l: the length parameter for the kernel
        :param sigma_f: the standard deviation given to the output of the
        black-box function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices
        :param X1: numpy.ndarray of shape (m, 1)
        :param X2: numpy.ndarray of shape (n, 1)
        :return: the covariance kernel matrix as a numpy.ndarray of
        shape (m, n)
        """
        first = np.sum(X1 ** 2, 1).reshape(-1, 1)
        second = np.sum(X2 ** 2, 1)
        sqdist = first + second - 2 * np.dot(X1, X2.T)
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation of points in a
        Gaussian process
        :param X_s: numpy.ndarray of shape (s, 1) containing all of the
        points whose mean and standard deviation should be calculated
        :return: mu, sigma
            mu is a numpy.ndarray of shape (s,) containing the mean for each
            point in X_s, respectively
            sigma is a numpy.ndarray of shape (s,) containing the standard
            deviation for each point in X_s, respectively
        """
        K = self.kernel(self.X, self.X)
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(K)

        mu_s = K_s.T.dot(K_inv).dot(self.Y).reshape(X_s.shape[0])

        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        var_s = np.diagonal(cov_s)

        return mu_s, var_s
