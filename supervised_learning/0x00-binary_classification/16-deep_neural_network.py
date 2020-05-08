#!/usr/bin/env python3
"""
Module to create a neural network
"""
import numpy as np


class DeepNeuralNetwork:
    """
    A class that defines a neural network with one hidden layer performing
    binary classification
    """

    def __init__(self, nx, layers):
        """
        class constructor
        :param nx: is the number of input features to the neuron
        :param layers: a list representing the number of nodes in each layer
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        # L is the number of layers in the neural network
        self.L = len(layers)
        # cache is a dictionary to hold all intermediary values of the network
        self.cache = {}
        # weights is a dictionary to hold all weighs and biased of the network
        weights = {}
        for i in range(len(layers)):
            if layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            key_w = 'W' + str(i + 1)
            key_b = 'b' + str(i + 1)
            if i == 0:
                weights[key_w] = np.random.randn(layers[i], nx)*np.sqrt(2 / nx)
            else:
                weights[key_w] = np.random.randn(layers[i], layers[
                    i-1]) * np.sqrt(2 / layers[i-1])
            weights[key_b] = np.zeros((layers[i], 1))
        self.weights = weights
