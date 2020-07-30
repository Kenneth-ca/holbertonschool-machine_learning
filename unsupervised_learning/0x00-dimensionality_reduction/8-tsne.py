#!/usr/bin/env python3
"""
Calculates the t-sne transformation
"""
import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Performs a t-SNE transformation
    :param X: numpy.ndarray of shape (n, d) containing the dataset to be
    transformed by t-SNE
        n is the number of data points
        d is the number of dimensions in each point
    :param ndims: is the new dimensional representation of X
    :param idims: the intermediate dimensional representation of X after PCA
    :param perplexity: the perplexity
    :param iterations: the number of iterations
    :param lr: the learning rate
    :return: Y, a numpy.ndarray of shape (n, ndim) containing the optimized
    low dimensional transformation of X
    """
    initial_momentum = 0.5
    final_momentum = 0.8
    min_gain = 0.1

    n, d = X.shape
    gains = np.ones((n, ndims))
    Y = np.random.randn(n, ndims)
    # Y - np.mean(Y, axis=0)
    iY = np.zeros((n, ndims))

    pca_result = pca(X, idims)
    P = P_affinities(pca_result, perplexity=perplexity)
    # Exaggeration
    P = 4 * P

    for i in range(iterations):
        dY, Q = grads(Y, P)
        if i < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        """gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain"""

        iY = momentum * iY - lr * dY
        Y = Y + iY - np.tile(np.mean(Y, 0), (n, 1))

        if (i + 1) % 100 == 0:
            C = cost(P, Q)
            print("Cost at iteration {}: {}".format(i + 1, C))

        # Remove exaggeration
        if (i + 1) == 100:
            P = P / 4

    return Y
