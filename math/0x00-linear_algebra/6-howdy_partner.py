#!/usr/bin/env python3
"""
Module that concatenates two arrays
"""


def cat_arrays(arr1, arr2):
    """
    Needs a matrix as input
    Returns a list of arrays concatenated
    """
    concated = arr1.copy()
    for i in arr2:
        concated.append(i)
    return concated
