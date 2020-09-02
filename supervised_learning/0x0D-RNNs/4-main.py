#!/usr/bin/env python3

import numpy as np
RNNCell = __import__('0-rnn_cell').RNNCell
deep_rnn = __import__('4-deep_rnn').deep_rnn

np.random.seed(1)
cell1 = RNNCell(10, 15, 1)
cell2 = RNNCell(15, 15, 1)
cell3 = RNNCell(15, 15, 5)
rnn_cells = [cell1, cell2, cell3]
for rnn_cell in rnn_cells:
    rnn_cell.bh = np.random.randn(1, 15)
cell3.by = np.random.randn(1, 5)
X = np.random.randn(6, 8, 10)
H_0 = np.zeros((3, 8, 15))
H, Y = deep_rnn(rnn_cells, X, H_0)
print(H.shape)
print(H)
print(Y.shape)
print(Y)
