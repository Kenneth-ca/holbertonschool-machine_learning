#!/usr/bin/env python3

import numpy as np
one_hot = __import__('3-one_hot').one_hot
load_model = __import__('9-model').load_model
predict = __import__('13-predict').predict


if __name__ == '__main__':
    datasets = np.load('../data/MNIST.npz')
    X_test = datasets['X_test']
    X_test = X_test.reshape(X_test.shape[0], -1)
    Y_test = datasets['Y_test']

    network = load_model('network2.h5')
    Y_pred = predict(network, X_test)
    print(Y_pred)
    print(np.argmax(Y_pred, axis=1))
    print(Y_test)