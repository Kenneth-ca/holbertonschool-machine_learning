#!/usr/bin/env python3
"""
Module to calculate the shape of a matrix
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
