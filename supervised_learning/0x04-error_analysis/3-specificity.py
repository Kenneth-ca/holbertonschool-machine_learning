#!/usr/bin/env python3
"""
Module to calculate specifity
"""
import numpy as np


def specificity(confusion):
    """
    a function that calculates specificity for each class in confusion matrix
    :param confusion: a confusion numpy.ndarray of shape (classes, classes)
    where row indices represent the correct labels and column indices
    represent the predicted labels
    :return: a numpy.ndarray of shape (classes,) containing the specificity
    of each class
    """
    # Specificity = True Negative(TN) / Negatives(N); N = TN + False
    # Positive(FP)
    FP = confusion.sum(axis=0) - np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TP = np.diag(confusion)
    TN = confusion.sum() - (FP + FN + TP)
    TNR = TN / (TN + FP)
    return TNR
