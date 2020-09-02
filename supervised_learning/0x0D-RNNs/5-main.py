#!/usr/bin/env python3

import numpy as np
BidirectionalCell = __import__('5-bi_forward'). BidirectionalCell

np.random.seed(5)
bi_cell =  BidirectionalCell(10, 15, 5)
print("Whf:", bi_cell.Whf)
print("Whb:", bi_cell.Whb)
print("Wy:", bi_cell.Wy)
print("bhf:", bi_cell.bhf)
print("bhb:", bi_cell.bhb)
print("by:", bi_cell.by)
bi_cell.bhf = np.random.randn(1, 15)
h_prev = np.random.randn(8, 15)
x_t = np.random.randn(8, 10)
h = bi_cell.forward(h_prev, x_t)
print(h.shape)
print(h)
