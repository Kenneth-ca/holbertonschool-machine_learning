#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Deep = __import__('3-deep_neural_network').DeepNeuralNetwork
one_hot_encode = __import__('0-one_hot_encode').one_hot_encode
one_hot_decode = __import__('1-one_hot_decode').one_hot_decode

lib = np.load('../data/MNIST.npz')
X_train_3D = lib['X_train']
Y_train = lib['Y_train']
X_valid_3D = lib['X_valid']
Y_valid = lib['Y_valid']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1)).T
Y_train_one_hot = one_hot_encode(Y_train, 10)
Y_valid_one_hot = one_hot_encode(Y_valid, 10)

deep = Deep.load('3-saved.pkl')
A_one_hot, cost = deep.train(X_train, Y_train_one_hot, iterations=100,
                             step=10, graph=False)
A = one_hot_decode(A_one_hot)
accuracy = np.sum(Y_train == A) / Y_train.shape[0] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))

A_one_hot, cost = deep.evaluate(X_valid, Y_valid_one_hot)
A = one_hot_decode(A_one_hot)
accuracy = np.sum(Y_valid == A) / Y_valid.shape[0] * 100
print("Validation cost:", cost)
print("Validation accuracy: {}%".format(accuracy))

deep.save('3-output')

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_valid_3D[i])
    plt.title(A[i])
    plt.axis('off')
plt.tight_layout()
plt.show()