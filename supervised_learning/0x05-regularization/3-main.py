#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
l2_reg_cost = __import__('2-l2_reg_cost').l2_reg_cost
l2_reg_create_layer = __import__('3-l2_reg_create_layer').l2_reg_create_layer

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    m = Y.shape[0]
    one_hot = np.zeros((m, classes))
    one_hot[np.arange(m), Y] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_train_3D = lib['X_train']
    Y_train = lib['Y_train']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1))
    Y_train_oh = one_hot(Y_train, 10)

    tf.set_random_seed(0)
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 10])
    h1 = l2_reg_create_layer(x, 256, tf.nn.tanh, 0.1)
    y_pred = l2_reg_create_layer(x, 10, None, 0.)
    cost = tf.losses.softmax_cross_entropy(y, y_pred)
    l2_cost = l2_reg_cost(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(l2_cost, feed_dict={x: X_train, y: Y_train_oh}))