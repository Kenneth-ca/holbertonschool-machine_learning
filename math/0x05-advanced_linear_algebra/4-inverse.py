#!/usr/bin/env python3
"""
Calculates the inverse of a matrix
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


def cofactor(matrix):
    """
    a function that calculates the cofactor of a matrix
    :param matrix: matrix is a list of lists whose minor matrix should be
    calculated
    :return: the cofactor matrix of a matrix
    """
    if type(matrix) is not list or not len(matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    for i in range(len(matrix)):
        if len(matrix) != len(matrix[i]):
            raise ValueError("matrix must be a non-empty square matrix")
        if type(matrix[i]) is not list or not len(matrix[i]):
            raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1:
        return [[1]]

    list_minor = []
    for i in range(len(matrix)):
        inner = []
        if i % 2 == 0:
            cof = 1
        else:
            cof = -1
        for j in range(len(matrix[0])):
            next_matrix = [x[:] for x in matrix]
            del next_matrix[i]
            for mat in next_matrix:
                del mat[j]
            determ = determinant(next_matrix) * cof
            inner.append(determ)
            cof = cof * -1
        list_minor.append(inner)

    return list_minor


def adjugate(matrix):
    """
    a function that calculates the adjugate
    :param matrix: a list of lists whose adjugate matrix should be calculated
    :return: the adjugate of a matrix
    """
    adj = cofactor(matrix)
    transpose = []
    for j in range(len(adj[0])):
        inner = []
        for i in range(len(adj)):
            inner.append(adj[i][j])
        transpose.append(inner)
    return transpose


def inverse(matrix):
    """
    a function that calculates the inverse of a matrix
    :param matrix: a list of lists whose inverse should be calculated
    :return: the inverse of matrix, or None if matrix is singular
    """
    adj = adjugate(matrix)
    deter = determinant(matrix)
    if deter == 0:
        return None
    inv = []
    for i in range(len(matrix)):
        inner = []
        for j in range(len(matrix[0])):
            inner.append(adj[i][j] / deter)
        inv.append(inner)
    return inv
