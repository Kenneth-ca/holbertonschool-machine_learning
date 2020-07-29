#!/usr/bin/env python3

import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost

np.random.seed(0)
X = np.loadtxt("mnist2500_X.txt")
X = pca(X, 50)
P = P_affinities(X)
Y = np.random.randn(X.shape[0], 2)
_, Q = grads(Y, P)
C = cost(P, Q)
print(C)
