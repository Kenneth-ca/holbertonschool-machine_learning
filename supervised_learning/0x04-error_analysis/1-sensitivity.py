#!/usr/bin/env python3
"""
Module to calculate sensivity
"""
import numpy as np


def sensitivity(confusion):
    """
    a function that calculates sensivirty for each class in confusion matrix
    :param confusion: a confusion numpy.ndarray of shape (classes, classes)
    where row indices represent the correct labels and column indices
    represent the predicted labels
    :return: a numpy.ndarray of shape (classes,) containing the sensitivity
    of each class
    """
    # Sensitivity = True Positive(TP) / Positive(P); P = TP+False Negative(FN)
    TP = np.diagonal(confusion)
    P = np.sum(confusion, axis=1)
    return TP / P
