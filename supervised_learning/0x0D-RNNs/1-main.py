#!/usr/bin/env python3

import numpy as np
RNNCell = __import__('0-rnn_cell').RNNCell
rnn = __import__('1-rnn').rnn

np.random.seed(1)
rnn_cell = RNNCell(10, 15, 5)
rnn_cell.bh = np.random.randn(1, 15)
rnn_cell.by = np.random.randn(1, 5)
X = np.random.randn(6, 8, 10)
h_0 = np.zeros((8, 15))
H, Y = rnn(rnn_cell, X, h_0)
print(H.shape)
print(H)
print(Y.shape)
print(Y)
