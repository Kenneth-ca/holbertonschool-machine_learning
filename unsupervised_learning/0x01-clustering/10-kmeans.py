#!/usr/bin/env python3
"""
Performs K-means on a dataset
"""
import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means on a dataset
    :param X: numpy.ndarray of shape (n, d) containing the dataset
    :param k: the number of clusters
    :return: C, clss
        C is a numpy.ndarray of shape (k, d) containing the centroid means for
        each cluster
        clss is a numpy.ndarray of shape (n,) containing the index of the
        cluster in C that each data point belongs to
    """
    k_means = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = k_means.cluster_centers_
    clss = k_means.labels_
    return C, clss
