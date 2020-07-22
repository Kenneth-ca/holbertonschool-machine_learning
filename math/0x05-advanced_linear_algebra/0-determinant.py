#!/usr/bin/env python3
"""
Calculates the determinant of a matrix
"""


def determinant(matrix):
    """
    a funciton that calculates the determinant of a matrix
    :param matrix: a list of lists whose determinant should be calculated
    :return: the determinant of matrix
    """
    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        return 1

    for i in range(len(matrix)):
        if len(matrix) != len(matrix[i]):
            raise ValueError("matrix must be a square matrix")
        if type(matrix[i]) is not list or not len(matrix[i]):
            raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]

    if len(matrix) == 2 and len(matrix[0]) == 2:
        return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])

    first_row = matrix[0]
    determ = 0
    cof = 1
    for i in range(len(matrix[0])):
        next_matrix = [x[:] for x in matrix]
        del next_matrix[0]
        for mat in next_matrix:
            del mat[i]
        determ += first_row[i] * determinant(next_matrix) * cof
        cof = cof * -1

    return determ
