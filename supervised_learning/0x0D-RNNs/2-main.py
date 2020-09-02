#!/usr/bin/env python3

import numpy as np
GRUCell = __import__('2-gru_cell').GRUCell

np.random.seed(2)
gru_cell = GRUCell(10, 15, 5)
print("Wz:", gru_cell.Wz)
print("Wr:", gru_cell.Wr)
print("Wh:", gru_cell.Wh)
print("Wy:", gru_cell.Wy)
print("bz:", gru_cell.bz)
print("br:", gru_cell.br)
print("bh:", gru_cell.bh)
print("by:", gru_cell.by)
gru_cell.bz = np.random.randn(1, 15)
gru_cell.br = np.random.randn(1, 15)
gru_cell.bh = np.random.randn(1, 15)
gru_cell.by = np.random.randn(1, 5)
h_prev = np.random.randn(8, 15)
x_t = np.random.randn(8, 10)
h, y = gru_cell.forward(h_prev, x_t)
print(h.shape)
print(h)
print(y.shape)
print(y)
