#!/usr/bin/env python3
"""
Module that concatenates two matrices
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Needs a matrix as input
    Returns the resulting matrix
    """
    return np.concatenate((mat1, mat2), axis=axis)
