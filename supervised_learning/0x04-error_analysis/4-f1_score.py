#!/usr/bin/env python3
"""
Module to calculate F1 score
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    a function that calculates F1 score for each class in confusion matrix
    :param confusion: a confusion numpy.ndarray of shape (classes, classes)
    where row indices represent the correct labels and column indices
    represent the predicted labels
    :return: a numpy.ndarray of shape (classes,) containing the F1 score
    of each class
    """
    sens = sensitivity(confusion)
    prec = precision(confusion)
    F1 = 2 * (prec * sens) / (prec + sens)
    return F1
