#!/usr/bin/env python3

import numpy as np
dropout_forward_prop = __import__('4-dropout_forward_prop').dropout_forward_prop
dropout_gradient_descent = __import__('5-dropout_gradient_descent').dropout_gradient_descent


def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    m = Y.shape[0]
    one_hot = np.zeros((classes, m))
    one_hot[Y, np.arange(m)] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_train_3D = lib['X_train']
    Y_train = lib['Y_train']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
    Y_train_oh = one_hot(Y_train, 10)

    np.random.seed(0)

    weights = {}
    weights['W1'] = np.random.randn(256, 784)
    weights['b1'] = np.zeros((256, 1))
    weights['W2'] = np.random.randn(128, 256)
    weights['b2'] = np.zeros((128, 1))
    weights['W3'] = np.random.randn(10, 128)
    weights['b3'] = np.zeros((10, 1))

    cache = dropout_forward_prop(X_train, weights, 3, 0.8)
    print(weights['W2'])
    dropout_gradient_descent(Y_train_oh, weights, cache, 0.1, 0.8, 3)
    print(weights['W2'])