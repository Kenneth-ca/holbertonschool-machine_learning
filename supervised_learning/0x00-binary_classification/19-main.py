#!/usr/bin/env python3

import numpy as np

Deep = __import__('19-deep_neural_network').DeepNeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
deep = Deep(X.shape[0], [5, 3, 1])
A, _ = deep.forward_prop(X)
cost = deep.cost(Y, A)
print(cost)