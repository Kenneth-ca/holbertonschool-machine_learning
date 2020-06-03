#!/usr/bin/env python3

import numpy as np
one_hot = __import__('3-one_hot').one_hot
load_model = __import__('9-model').load_model
test_model = __import__('12-test').test_model


if __name__ == '__main__':
    datasets = np.load('../data/MNIST.npz')
    X_test = datasets['X_test']
    X_test = X_test.reshape(X_test.shape[0], -1)
    Y_test = datasets['Y_test']
    Y_test_oh = one_hot(Y_test)

    network = load_model('network2.h5')
    print(test_model(network, X_test, Y_test_oh))
