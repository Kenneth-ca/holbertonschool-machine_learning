#!/usr/bin/env python3
"""
Module to create a neural network
"""
import numpy as np


class NeuralNetwork:
    """
    A class that defines a neural network with one hidden layer performing
    binary classification
    """

    def __init__(self, nx, nodes):
        """
        class constructor
        :param nx: is the number of input features to the neuron
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        # hidden layer
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        # output neuron
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        :param X: np array with the input data of shape (nx, m)
        :return: the private attributes __A1 and __A2
        """
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))
        return self.__A1, self.__A2

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
        evaluates the neural network prediction
        :param X: np array with input data of shape (nx, m)
        :param Y: np array with correct label of shape (1, m)
        :return: neuron´s prediction and cost of the network
        """
        self.forward_prop(X)
        prediction = np.where(self.__A2 >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A2)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        calculates one pass of gradient descent on the neuron
        :param X: np array with input data of shape (nx, m)
        :param Y: np array with correct labels of shape (1, m)
        :param A1: np array with activated hidden layer output of shape (1, m)
        :param A2: np array with activated output of shape (1, m)
        :param alpha: the learning rate
        :return: no return
        """
        # gradient descent for hidden layer
        dz2 = A2 - Y
        dw2 = np.matmul(A1, dz2.T) / A1.shape[1]
        db2 = np.sum(dz2, axis=1, keepdims=True) / A2.shape[1]

        # derivative of the sigmoid function
        da1 = A1 * (1 - A1)
        # gradient descent for output layer
        dz1 = np.matmul(self.__W2.T, dz2)
        dz1 = dz1 * da1
        dw1 = np.matmul(X, dz1.T) / A1.shape[1]
        db1 = np.sum(dz1, axis=1, keepdims=True) / A1.shape[1]
        # updated value for weights and bias
        self.__W2 = self.__W2 - alpha * dw2.T
        self.__b2 = self.__b2 - alpha * db2
        self.__W1 = self.__W1 - alpha * dw1.T
        self.__b1 = self.__b1 - alpha * db1

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2
