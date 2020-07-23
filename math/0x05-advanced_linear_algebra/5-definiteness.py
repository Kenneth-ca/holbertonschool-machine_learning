#!/usr/bin/env python3
"""
Calculates the definiteness of a matrix
"""
import numpy as np


def definiteness(matrix):
    """
    a function that calculates the definitiness of a matrix
    :param matrix: a numpy.ndarray of shape (n, n) whose definiteness should
    be calculated
    :return: if the matrix is positive definite, positive semi-definite,
    negative semi-definite, negative definite of indefinite, respectively
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) != 2:
        return None
    row, col = matrix.shape
    if row != col:
        return None
    # Matrix has to be symmetric to calculate definiteness
    if not (matrix == matrix.T).all():
        return None
    eigenvalues = np.linalg.eigvals(matrix)

    pos = 0
    neg = 0
    semi = 0
    for i in eigenvalues:
        if i > 0:
            pos = 1
        if i < 0:
            neg = 1
        if i == 0:
            semi = 1

    if pos and not semi and not neg:
        return "Positive definite"
    elif pos and semi and not neg:
        return "Positive semi-definite"
    elif not pos and not semi and neg:
        return "Negative definite"
    elif not pos and semi and neg:
        return "Negative semi-definite"
    elif pos and not semi and neg:
        return "Indefinite"
    else:
        return None
