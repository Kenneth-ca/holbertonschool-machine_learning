#!/usr/bin/env python3

import numpy as np

NN = __import__('11-neural_network').NeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X.shape[0], 3)
_, A = nn.forward_prop(X)
cost = nn.cost(Y, A)
print(cost)