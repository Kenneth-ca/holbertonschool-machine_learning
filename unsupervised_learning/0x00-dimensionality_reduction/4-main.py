#!/usr/bin/env python3

import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities

X = np.loadtxt("mnist2500_X.txt")
X = pca(X, 50)
P = P_affinities(X)
print('P:', P.shape)
print(P)
print(np.sum(P))
