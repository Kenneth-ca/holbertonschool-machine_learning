#!/usr/bin/env python3
"""
Module to create a neuron
"""
import numpy as np


class Neuron:
    """
    A class that defines a single neuron
    """
    W = np.random.randn(1, 784)
    b = 0
    A = 0

    def __init__(self, nx):
        """
        class constructor
        :param nx: is the number of input features to the neuron
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")