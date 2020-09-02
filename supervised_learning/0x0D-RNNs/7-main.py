#!/usr/bin/env python3

import numpy as np
BidirectionalCell = __import__('7-bi_output'). BidirectionalCell

np.random.seed(7)
bi_cell =  BidirectionalCell(10, 15, 5)
bi_cell.by = np.random.randn(1, 5)
H = np.random.randn(6, 8, 30)
Y = bi_cell.output(H)
print(Y.shape)
print(Y)
