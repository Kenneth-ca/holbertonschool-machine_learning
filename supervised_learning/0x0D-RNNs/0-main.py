#!/usr/bin/env python3

import numpy as np
RNNCell = __import__('0-rnn_cell').RNNCell

np.random.seed(0)
rnn_cell = RNNCell(10, 15, 5)
print("Wh:", rnn_cell.Wh)
print("Wy:", rnn_cell.Wy)
print("bh:", rnn_cell.bh)
print("by:", rnn_cell.by)
rnn_cell.bh = np.random.randn(1, 15)
rnn_cell.by = np.random.randn(1, 5)
h_prev = np.random.randn(8, 15)
x_t = np.random.randn(8, 10)
h, y = rnn_cell.forward(h_prev, x_t)
print(h.shape)
print(h)
print(y.shape)
print(y)
