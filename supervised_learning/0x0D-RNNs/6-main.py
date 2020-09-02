#!/usr/bin/env python3

import numpy as np
BidirectionalCell = __import__('6-bi_backward'). BidirectionalCell

np.random.seed(6)
bi_cell =  BidirectionalCell(10, 15, 5)
bi_cell.bhb = np.random.randn(1, 15)
h_next = np.random.randn(8, 15)
x_t = np.random.randn(8, 10)
h = bi_cell.backward(h_next, x_t)
print(h.shape)
print(h)
