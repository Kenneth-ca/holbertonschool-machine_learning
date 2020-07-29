#!/usr/bin/env python3
"""
Calculates the symmetric P affinities
"""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a data set
    :param X: numpy.ndarray of shape (n, d) containing the dataset to be
    transformed by t-SNE
        n is the number of data points
        d is the number of dimensions in each point
    :param tol: the maximum tolerance allowed (inclusive) for the difference
    in Shannon entropy from perplexity for all Gaussian distributions
    :param perplexity: perplexity that all Gaussian distributions should have
    :return: P, a numpy.ndarray of shape (n, n) containing the symmetric P
    affinities
    """
    n, d = X.shape
    D, P, betas, H = P_init(X, perplexity)

    if n == 0:
        return P

    for i in range(n):
        copy = D[i].copy()
        copy = np.delete(copy, i, axis=0)
        Hi, Pi = HP(copy, betas[i])
        Hdiff = Hi - H

        low = None
        high = None

        while np.abs(Hdiff) > tol:
            if Hdiff > 0:
                low = betas[i, 0]
                if high is None:
                    betas[i] = betas[i] * 2
                else:
                    betas[i] = (betas[i] + high) / 2

            else:
                high = betas[i, 0]
                if low is None:
                    betas[i] = betas[i] / 2
                else:
                    betas[i] = (betas[i] + low) / 2

            Hi, Pi = HP(copy, betas[i])
            Hdiff = Hi - H
        Pi = np.insert(Pi, i, 0)
        P[i] = Pi
    P = (P.T + P) / (2 * n)
    return P
