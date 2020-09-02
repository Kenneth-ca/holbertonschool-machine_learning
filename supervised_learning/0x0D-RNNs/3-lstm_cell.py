#!/usr/bin/env python3
"""
Creates the class LSTMCell
"""
import numpy as np


class LSTMCell:
    """
    Represents a gated recurrent unit
    """

    def __init__(self, i, h, o):
        """
        Initializes class LSTMCell
        :param i: the dimensionality of the data
        :param h: the dimensionality of the hidden state
        :param o: the dimensionality of the outputs
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
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
        :return: h_next, c_next, y
            h_next is the next hidden state
            c_next is the next cell state
            y is the output of the cell
        """
        # Concat h_prev and x_t to match Wh dimensions
        x_concat0 = np.concatenate((h_prev, x_t), axis=1)
        f = np.matmul(x_concat0, self.Wf) + self.bf
        f = 1 / (1 + np.exp(-f))
        u = np.matmul(x_concat0, self.Wu) + self.bu
        u = 1 / (1 + np.exp(-u))
        c = np.matmul(x_concat0, self.Wc) + self.bc
        c = np.tanh(c)
        o = np.matmul(x_concat0, self.Wo) + self.bo
        o = 1 / (1 + np.exp(-o))

        c_next = (u * c) + (f * c_prev)
        h_t = o * np.tanh(c_next)

        y = np.matmul(h_t, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_t, c_next, y
