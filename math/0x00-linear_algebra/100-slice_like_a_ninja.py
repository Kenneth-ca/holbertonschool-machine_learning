#!/usr/bin/env python3
"""
Module that multiplies two matrices
"""


def np_slice(matrix, axes={}):
    """
    Needs a matrix as input
    Returns the resulting matrix
    """
    sliced = []
    max_key = max(axes)
    for i in range(max_key + 1):
        if i in axes.keys():
            sliced.append(slice(*axes.get(i)))
        else:
            sliced.append(slice(None, None, None))
    return matrix[tuple(sliced)]
