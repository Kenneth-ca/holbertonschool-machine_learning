#!/usr/bin/env python3
"""
Module that preforms aritmetic operations to matrices
"""


def np_elementwise(mat1, mat2):
    """
    Needs a matrix as input
    Returns the resulting matrix
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
