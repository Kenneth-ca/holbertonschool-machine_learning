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
    rows = 0
    colums = 0
    ele = 0
    # Exists
    if type(matrix) != list:
        return [0]
    try

    for row in matrix:
        rows += 1
        # Has rows
        if type(i) == list:
            for col in row:
                colums += 1
                # Has columns

        else:
            return(0)
    integers.append(ele)
    return integers
