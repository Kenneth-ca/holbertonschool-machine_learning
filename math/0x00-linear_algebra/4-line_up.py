#!/usr/bin/env python3
"""
Module the sum of two arrays
"""


def add_arrays(arr1, arr2):
    """
    Needs a matrix as input
    Returns the sum of two arrays a a list
    """
    if len(arr1) != len(arr2):
        return None
    return [(arr1[i] + arr2[i]) for i in range(len(arr1))]
