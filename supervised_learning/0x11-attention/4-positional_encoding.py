#!/usr/bin/env python3
"""
Creates the function positional encoding
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer
    :param max_seq_len: an integer representing the maximum sequence length
    :param dm: the model depth
    :return: numpy.ndarray of shape (max_seq_len, dm) containing the
    positional encoding vectors
    """
    positional_embeddings = np.zeros((max_seq_len, dm))

    for position in range(max_seq_len):
        for i in range(0, dm, 2):
            div = np.exp(i * -np.log(10000.0) / dm)
            positional_embeddings[position, i] = (
                np.sin(position * div))
            positional_embeddings[position, i + 1] = (
                np.cos(position * div))

    return positional_embeddings
