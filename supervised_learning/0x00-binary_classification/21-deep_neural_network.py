#!/usr/bin/env python3
"""
Module to create a deep neural network
"""
import numpy as np


class DeepNeuralNetwork:
    """
    A class that defines a deep neural network with one hidden layer performing
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
        self.__L = len(layers)
        # cache is a dictionary to hold all intermediary values of the network
        self.__cache = {}
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
        self.__weights = weights

    def forward_prop(self, X):
        """
        calculates the forward propagation of the deep neural network
        :param X:np array with the input data of shape (nx, m)
        :return: the output of the deep neural network and the cache,
        where cache is the activated output of each layer
        """
        # Input layer
        self.__cache['A0'] = X
        # Hidden and output layer
        for i in range(self.__L):
            # create keys to access weights(W), biases(b) and store in cache
            key_w = 'W' + str(i + 1)
            key_b = 'b' + str(i + 1)
            key_cache = 'A' + str(i + 1)
            key_cache_last = 'A' + str(i)
            # store activation in cache
            output_Z = np.matmul(self.__weights[key_w], self.__cache[
                key_cache_last]) + self.__weights[key_b]
            output_A = 1 / (1 + np.exp(-output_Z))
            self.__cache[key_cache] = output_A
        return output_A, self.__cache

    def cost(self, Y, A):
        """
        calculates the cost of the model using logistic regression
        :param Y: a np array with correct labels of shape (1, m)
        :param A: a np array with the activated output of shape (1, m)
        :return: the cost
        """
        cost = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = np.sum(cost)
        cost = - cost / A.shape[1]
        return cost

    def evaluate(self, X, Y):
        """
        evaluates the deep neural network prediction
        :param X: np array with input data of shape (nx, m)
        :param Y: np array with correct label of shape (1, m)
        :return: neuron´s prediction and cost of the network
        """
        A, _ = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        calculates one pass of gradient descent on the neuron
        :param Y: np array with correct labels of shape (1, m)
        :param cache: dictionary containing all intermediary values of the
        network
        :param alpha: the learning rate
        :return: no return
        """
        m = Y.shape[1]
        for i in reversed(range(self.__L)):
            # create keys to access weights(W), biases(b) and store in cache
            key_w = 'W' + str(i + 1)
            key_b = 'b' + str(i + 1)
            key_cache = 'A' + str(i + 1)
            key_cache_dw = 'A' + str(i)
            # Activation
            A = cache[key_cache]
            A_dw = cache[key_cache_dw]
            if i == self.__L - 1:
                dz = A - Y
                W = self.__weights[key_w]
            else:
                da = A * (1 - A)
                dz = np.matmul(W.T, dz)
                dz = dz * da
                W = self.__weights[key_w]
            dw = np.matmul(A_dw, dz.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            self.__weights[key_w] = self.__weights[key_w] - alpha * dw.T
            self.__weights[key_b] = self.__weights[key_b] - alpha * db

    @property
    def cache(self):
        """
        attribute getter for cache
        :return: cache
        """
        return self.__cache

    @property
    def L(self):
        """
        attribute getter for L (number of layers)
        :return: L
        """
        return self.__L

    @property
    def weights(self):
        """
        attribute getter for weights
        :return: weights
        """
        return self.__weights
