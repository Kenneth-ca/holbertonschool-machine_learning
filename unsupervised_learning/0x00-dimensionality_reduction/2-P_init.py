#!/usr/bin/env python3
"""
Initializes all variables required to calculate the P affinities in t-SNE
"""
import numpy as np


def P_init(X, perplexity):
    """
    a function that initialize variables for t-SNE
    :param X: numpy.ndarray of shape (n, d) containing the dataset to be
    transformed by t-SNE
        n is the number of data points
        d is the number of dimensions in each point
    :param perplexity: is the perplexity that all Gaussian distributions
    should have
    :return: (D, P, betas, H)
        D: a numpy.ndarray of shape (n, n) that calculates the pairwise
        distance between two data points
        P: a numpy.ndarray of shape (n, n) initialized to all 0‘s that will
        contain the P affinities
        betas: a numpy.ndarray of shape (n, 1) initialized to all 1’s that will
        contain all of the beta values
        H is the Shannon entropy for perplexity perplexity
    """
    n, d = X.shape

    # This method uses cuadratic expansion (x - y)^2 = x^2 - 2xy + y^2
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    np.fill_diagonal(D, 0)

    # This method adds a new axis to allow broadcasting and have each the
    # distance between each dimension
    # example shapes: X1 (1, 2500, 50) and X2 (2500, 1, 50)
    # X1 = X[np.newaxis, :, :]
    # X2 = X[:, np.newaxis, :]
    # D = np.sum(np.square(X1 - X2), axis=2)

    P = np.zeros((n, n))

    H = np.log2(perplexity)

    betas = np.ones((n, 1))

    return D, P, betas, H
