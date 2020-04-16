#!/usr/bin/env python3
"""
Module that concatenates two matrices
"""


def matrix_shape(matrix):
    """
    Needs a matrix as input
    Returns the shape as a list of integers
    """
    if type(matrix[0]) is not list:
        return [len(matrix)]
    else:
        return [len(matrix)] + matrix_shape(matrix[0])


def concat_recursive(mat1, mat2, axe):
    """
    Needs a matrix as input
    Returns a concatenated matrix
    """
    result = []
    if axe == 0:
        result = mat1 + mat2
        return result
    for i in range(len(mat1)):
        result.append(concat_recursive(mat1[i], mat2[i], axe - 1))
    return result


def cat_matrices(mat1, mat2, axis=0):
    """
    Needs a matrix as input
    Returns the resulting matrix
    """
    if len(matrix_shape(mat1)) > axis and len(matrix_shape(mat2)) > axis:
        return concat_recursive(mat1, mat2, axis)
    else:
        return None
