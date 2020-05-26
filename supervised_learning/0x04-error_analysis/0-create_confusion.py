#!/usr/bin/env python3
"""
Module to creates a confusion matrix
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    a function that creates a confusion matrix
    :param labels: one-hot numpy.ndarray of shape (m, classes) containing the
    correct labels for each data point
        m is the number of data points
        classes is the number of classes
    :param logits: one-hot numpy.ndarray of shape (m, classes) containing the
    predicted labels
    :return: a confusion numpy.ndarray of shape (classes, classes) with row
    indices representing the correct labels and column indices representing
    the predicted labels
    """
    return np.matmul(labels.T, logits)
