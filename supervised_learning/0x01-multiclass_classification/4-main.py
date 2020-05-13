#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Deep3 = __import__('3-deep_neural_network').DeepNeuralNetwork
Deep4 = __import__('4-deep_neural_network').DeepNeuralNetwork
one_hot_encode = __import__('0-one_hot_encode').one_hot_encode
one_hot_decode = __import__('1-one_hot_decode').one_hot_decode

lib= np.load('../data/MNIST.npz')
X_train_3D = lib['X_train']
Y_train = lib['Y_train']
X_valid_3D = lib['X_valid']
Y_valid = lib['Y_valid']
X_test_3D = lib['X_test']
Y_test = lib['Y_test']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1)).T
X_test = X_test_3D.reshape((X_test_3D.shape[0], -1)).T
Y_train_one_hot = one_hot_encode(Y_train, 10)
Y_valid_one_hot = one_hot_encode(Y_valid, 10)
Y_test_one_hot = one_hot_encode(Y_test, 10)

print('Sigmoid activation:')
deep3 = Deep3.load('3-output.pkl')
A_one_hot3, cost3 = deep3.evaluate(X_train, Y_train_one_hot)
A3 = one_hot_decode(A_one_hot3)
accuracy3 = np.sum(Y_train == A3) / Y_train.shape[0] * 100
print("Train cost:", cost3)
print("Train accuracy: {}%".format(accuracy3))
A_one_hot3, cost3 = deep3.evaluate(X_valid, Y_valid_one_hot)
A3 = one_hot_decode(A_one_hot3)
accuracy3 = np.sum(Y_valid == A3) / Y_valid.shape[0] * 100
print("Validation cost:", cost3)
print("Validation accuracy: {}%".format(accuracy3))
A_one_hot3, cost3 = deep3.evaluate(X_test, Y_test_one_hot)
A3 = one_hot_decode(A_one_hot3)
accuracy3 = np.sum(Y_test == A3) / Y_test.shape[0] * 100
print("Test cost:", cost3)
print("Test accuracy: {}%".format(accuracy3))

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_test_3D[i])
    plt.title(A3[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

print('\nTanh activaiton:')

deep4 = Deep4.load('4-saved.pkl')
A_one_hot4, cost4 = deep4.train(X_train, Y_train_one_hot, iterations=100,
                                step=10, graph=False)
A4 = one_hot_decode(A_one_hot4)
accuracy4 = np.sum(Y_train == A4) / Y_train.shape[0] * 100
print("Train cost:", cost4)
print("Train accuracy: {}%".format(accuracy4))
A_one_hot4, cost4 = deep4.evaluate(X_valid, Y_valid_one_hot)
A4 = one_hot_decode(A_one_hot4)
accuracy4 = np.sum(Y_valid == A4) / Y_valid.shape[0] * 100
print("Validation cost:", cost4)
print("Validation accuracy: {}%".format(accuracy4))
A_one_hot4, cost4 = deep4.evaluate(X_test, Y_test_one_hot)
A4 = one_hot_decode(A_one_hot4)
accuracy4 = np.sum(Y_test == A4) / Y_test.shape[0] * 100
print("Test cost:", cost4)
print("Test accuracy: {}%".format(accuracy4))
deep4.save('4-output')

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_test_3D[i])
    plt.title(A4[i])
    plt.axis('off')
plt.tight_layout()
plt.show()