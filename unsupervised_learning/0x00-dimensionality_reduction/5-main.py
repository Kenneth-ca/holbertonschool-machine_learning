#!/usr/bin/env python3

import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities

np.random.seed(0)
Y = np.random.randn(2500, 2)
Q, num = Q_affinities(Y)
print('num:', num.shape)
print(num)
print(np.sum(num))
print('Q:', Q.shape)
print(Q)
print(np.sum(Q))
