#!/usr/bin/env python3
"""
Creates the class LSTMCell
"""
import numpy as np


class BidirectionalCell:
    """
    Represents a bidirectional cell of an RNN
    """

    def __init__(self, i, h, o):
        """

        :param i: the dimensionality of the data
        :param h: the dimensionality of the hidden state
        :param o: the dimensionality of the outputs
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(2 * h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step
        :param h_prev: numpy.ndarray of shape (m, h) containing the previous
        hidden state
            Returns: h_next, y
            h_next is the next hidden state
            y is the output of the cell
        :param x_t: numpy.ndarray of shape (m, i) that contains the data input
        for the cell
            m is the batch size for the data
        :return: h_next
            h_next is the next hidden state
        """
        # Concat h_prev and x_t to match Wh dimensions
        x_concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.matmul(x_concat, self.Whf) + self.bhf
        h_next = np.tanh(h_next)

        return h_next

    def backward(self, h_next, x_t):
        """
        Performs forward propagation for one time step
        :param h_prev: numpy.ndarray of shape (m, h) containing the previous
        hidden state
            Returns: h_next, y
            h_next is the next hidden state
            y is the output of the cell
        :param x_t: numpy.ndarray of shape (m, i) that contains the data input
        for the cell
            m is the batch size for the data
        :return: h_next
            h_next is the next hidden state
        """
        # Concat h_prev and x_t to match Wh dimensions
        x_concat = np.concatenate((h_next, x_t), axis=1)
        h_back = np.matmul(x_concat, self.Whb) + self.bhb
        h_back = np.tanh(h_back)

        return h_back

    def output(self, H):
        """
        Calculates all outputs for the RNN
        :param H: numpy.ndarray of shape (t, m, 2 * h) that contains the
        concatenated hidden states from both directions, excluding their
        initialized states
            t is the number of time steps
            m is the batch size for the data
            h is the dimensionality of the hidden states
        :return: Y, the outputs
        """
        t, m, h_2 = H.shape
        o = self.by.shape[-1]
        Y = np.zeros((t, m, o))

        for step in range(t):
            Y[step] = np.matmul(H[step], self.Wy) + self.by
            Y[step] = np.exp(Y[step]) / np.sum(np.exp(Y[step]),
                                               axis=1,
                                               keepdims=True)
        return Y
