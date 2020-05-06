#!/usr/bin/env python3
"""
Module to create a neuron
"""
import numpy as np


class Neuron:
    """
    A class that defines a single neuron
    """

    def __init__(self, nx):
        """
        class constructor
        :param nx: is the number of input features to the neuron
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        getter function for W
        :return: weight vector neuron
        """
        return self.__W

    @property
    def b(self):
        """
        getter function for b
        :return: bias for neuron
        """
        return self.__b

    @property
    def A(self):
        """
        getter function for W
        :return: activated output of the neuron
        """
        return self.__A
