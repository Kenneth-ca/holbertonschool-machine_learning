#!/usr/bin/env python3
"""
Loads data from a file as a pd.DataFrame
"""
import pandas as pd


def from_file(filename, delimiter):
    """
    Loads from a file as a dataframe
    :param filename: is the file to load from
    :param delimiter: is the column separator
    :return: the loaded dataframe
    """
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
