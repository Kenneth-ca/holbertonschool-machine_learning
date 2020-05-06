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

    def forward_prop(self, X):
        """
        calculates the forward propagation of the neuron
        :param X: a np array with shape (nx, m) that contains the input data
        :return: private attribute __A
        """
        preactivation = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-preactivation))
        return self.__A

    def cost(self, Y, A):
        """
        calculates the cost of the model using logistic regression
        :param Y: a np array with shape (1, m) with correct labels
        :param A: a np array with shape (1, m) containing the activated output
        :return: the cost
        """
        cost = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = np.sum(cost)
        cost = - cost / A.shape[1]
        return cost

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
