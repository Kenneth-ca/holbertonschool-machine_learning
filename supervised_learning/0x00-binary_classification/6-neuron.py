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
        :param X: a np array with the input data of shape (nx, m)
        :return: private attribute __A
        """
        preactivation = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-preactivation))
        return self.__A

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
        evaluates the neuron prediction
        :param X: np array with input data of shape (nx, m)
        :param Y: np array with correct label of shape (1, m)
        :return: neuron´s prediction and cost of the network
        """
        self.forward_prop(X)
        prediction = np.where(self.__A >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        calculates one pass of gradient descent on the neuron
        :param X: np array with input data of shape (nx, m)
        :param Y: np array with correct labels of shape (1, m)
        :param A: np array with activated output of shape (1, m)
        :param alpha: the learning rate
        :return: no return
        """
        dz = A - Y
        dw = np.matmul(X, dz.T) / A.shape[1]
        db = np.sum(dz) / A.shape[1]
        self.__W = self.__W - alpha * dw.T
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        trains the neuron
        :param X: np array with input data of shape (nx, m)
        :param Y: np array with correct labels of shape (1, m)
        :param iterations: iterations of the training
        :param alpha: learning rate
        :return: the evaluation of the training data
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            activations = self.forward_prop(X)
            self.gradient_descent(X, Y, activations, alpha)
        return self.evaluate(X, Y)

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
