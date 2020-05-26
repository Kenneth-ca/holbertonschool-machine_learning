#!/usr/bin/env python3
"""
Updates the weights and biases of a neural network using gradient descent
with L2 regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    a function that updates w and b of a NN using gradient descent with L2
    regularization
    :param Y: one-hot numpy.ndarray of shape (classes, m) that contains the
    correct labels for the data
        classes is the number of classes
        m is the number of data points
    :param weights: a dictionary of the weights and biases of the neural
    network
    :param cache: a dictionary of the outputs of each layer of the neural
    network
    :param alpha: the learning rate
    :param lambtha: the L2 regularization parameter
    :param L: the number of layers of the network
    :return: no return
    """
    m = Y.shape[1]
    for i in reversed(range(L)):
        # create keys to access weights(W), biases(b) and store in cache
        key_w = 'W' + str(i + 1)
        key_b = 'b' + str(i + 1)
        key_cache = 'A' + str(i + 1)
        key_cache_dw = 'A' + str(i)
        # Activation
        A = cache[key_cache]
        A_dw = cache[key_cache_dw]
        if i == L - 1:
            dz = A - Y
            W = weights[key_w]
        else:
            da = 1 - (A * A)
            dz = np.matmul(W.T, dz)
            dz = dz * da
            W = weights[key_w]
        dw = np.matmul(A_dw, dz.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        weights[key_w] = weights[key_w] - alpha * (dw.T + (lambtha / m *
                                                           weights[key_w]))
        weights[key_b] = weights[key_b] - alpha * db
