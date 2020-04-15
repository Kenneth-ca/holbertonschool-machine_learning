#!/usr/bin/env python3
"""
Module the sum of two matrices
"""


def add_matrices2D(mat1, mat2):
    """
    Needs a matrix as input
    Returns the sum of two matrices a a list
    """
    if len(mat1) != len(mat2):
        return None
    if len(mat1[0]) != len(mat2[0]):
        return None
    return [[(mat1[i][j] + mat2[i][j]) for j in range(len(mat1[0]))]
            for i in range(len(mat1))]
