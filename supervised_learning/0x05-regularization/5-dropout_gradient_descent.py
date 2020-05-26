#!/usr/bin/env python3
"""
Conducts gradient descent using Dropout:
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    a function that conducts gradient descent using dropout
    :param Y: one-hot numpy.ndarray of shape (classes, m) that contains the
    correct labels for the data
        classes is the number of classes
        m is the number of data points
    :param weights: a dictionary of the weights and biases of the neural
    network
    :param cache: a dictionary of the outputs and dropout masks of each layer
    :param alpha: the learning rate
    :param keep_prob: the probability that a node will be kept
    :param L: the number of layers in the network
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
            dz = dz * da * cache["D{}".format(i + 1)]
            dz = dz / keep_prob
            W = weights[key_w]
        dw = np.matmul(A_dw, dz.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        weights[key_w] = weights[key_w] - alpha * dw.T
        weights[key_b] = weights[key_b] - alpha * db
