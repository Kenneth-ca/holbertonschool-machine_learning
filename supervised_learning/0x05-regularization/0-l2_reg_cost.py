#!/usr/bin/env python3
"""
Calculates the cost of a neural network with L2 regularization:
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    a function that calculates the cost of a NN with L2 regularization
    :param cost: the cost of the network without L2 regularization
    :param lambtha: the regularization parameter
    :param weights: a dictionary of the weights and biases (numpy.ndarrays)
    of the neural network
    :param L: the number of layers in the neural network
    :param m: the number of data points used
    :return: the cost of the network accounting for L2 regularization
    """
    L2 = 0
    for i in range(L):
        w = weights["W{}".format(i + 1)]
        norm = np.linalg.norm(w)
        L2 += (lambtha / 2 / m * norm)
    return L2 + cost
