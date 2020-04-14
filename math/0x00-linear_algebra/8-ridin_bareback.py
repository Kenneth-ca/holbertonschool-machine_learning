#!/usr/bin/env python3
"""
Module that multiplies two matrices
"""


def mat_mul(mat1, mat2):
    """
    Needs a matrix as input
    Returns a new matrix or None if cannot be multiplied
    """
    if len(mat1[0]) == len(mat2):
        new = []
        for row1 in range(len(mat1)):
            inner = []
            for col2 in range(len(mat2[0])):
                number = 0
                for col1 in range(len(mat1[0])):
                    number += (mat1[row1][col1] * mat2[col1][col2])
                inner.append(number)
            new.append(inner)
        return new
    else:
        return None
