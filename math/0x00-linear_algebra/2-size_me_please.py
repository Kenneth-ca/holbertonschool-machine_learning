#!/usr/bin/env python3
"""
Module to calculate the shape of a matrix
"""

def matrix_shape(matrix):
    """
    Needs a matrix as input
    Returns the shape as a list of integers
    """
    integers = []
    #Check for list
    if type(matrix) != list:
        return [0]
    # Number of rows/elements
    try:
        integers.append(len(matrix))
    except:
        return [0]
    # Number of columns/elements
    try:
        integers.append(len(matrix[0]))
    except:
        pass
    # Number of elements
    try:
        integers.append(len(matrix[0][0]))
    except:
        pass
    return integers
