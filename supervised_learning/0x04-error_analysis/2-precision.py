#!/usr/bin/env python3
"""
Module to calculate precision
"""
import numpy as np


def precision(confusion):
    """
    a function that calculates precision for each class in confusion matrix
    :param confusion: a confusion numpy.ndarray of shape (classes, classes)
    where row indices represent the correct labels and column indices
    represent the predicted labels
    :return: a numpy.ndarray of shape (classes,) containing the precision
    of each class
    """
    # Precision = TP / (TP + FP)
    TP = np.diagonal(confusion)
    TP_FP = np.sum(confusion, axis=0)
    return TP / TP_FP
