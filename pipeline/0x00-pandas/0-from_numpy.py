#!/usr/bin/env python3
"""
Creates a pd.DataFrame from a np.ndarray
"""
import pandas as pd


def from_numpy(array):
    """
    Creates a dataframe
    :param array: the np.ndarray from which you should create the pd.DataFrame
    :return: the newly created pd.DataFrame
    """
    pd.set_option('display.max_columns', None)
    df = pd.DataFrame(array)
    _, n = array.shape
    alphabet = range(65, 65 + n)
    alphabet = list(map(lambda x: chr(x), alphabet))
    df.columns = alphabet
    return df
