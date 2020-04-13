#!/usr/bin/env python3
"""
Module that computes the traspose of a 2D matrix
"""


def matrix_transpose(matrix):
    """
    Needs a matrix as input
    Returns the traspose of a 2D matrix
    """
    tras = []
    for row in range(len(matrix[0])):
        tras.append([matrix[col][row] for col in range(len(matrix))])
    return tras
