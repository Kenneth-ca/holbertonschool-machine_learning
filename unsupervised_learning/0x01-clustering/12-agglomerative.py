#!/usr/bin/env python3
"""
Performs agglomerative clustering on a dataset
"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering on a dataset
    :param X: numpy.ndarray of shape (n, d) containing the dataset
    :param dist: maximum cophenetic distance for all clusters
    :return: clss, a numpy.ndarray of shape (n,) containing the cluster
    indices for each data point
    """
    hierarchy = scipy.cluster.hierarchy
    linkage = hierarchy.linkage(X, method='ward')
    fcluster = hierarchy.fcluster(linkage, dist, criterion='distance')

    hierarchy.dendrogram(linkage, color_threshold=dist)
    plt.figure()
    plt.show()

    return fcluster
