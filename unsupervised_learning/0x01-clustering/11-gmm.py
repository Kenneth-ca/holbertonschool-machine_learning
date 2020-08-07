#!/usr/bin/env python3
"""
Calculates a GMM from a dataset
"""
import sklearn.mixture


def gmm(X, k):
    """
    Calculates a GMM from a dataset
    :param X: numpy.ndarray of shape (n, d) containing the dataset
    :param k: the number of clusters
    :return: pi, m, S, clss, bic
    """
    model = sklearn.mixture.GaussianMixture(n_components=k).fit(X)

    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)

    return pi, m, S, clss, bic
