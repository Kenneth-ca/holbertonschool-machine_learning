#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
learning_rate_decay = __import__('12-learning_rate_decay').learning_rate_decay

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    one_hot = np.zeros((Y.shape[0], classes))
    one_hot[np.arange(Y.shape[0]), Y] = 1
    return one_hot

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_3D = lib['X_train']
    Y = lib['Y_train']
    X = X_3D.reshape((X_3D.shape[0], -1))
    Y_oh = one_hot(Y, 10)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./graph.ckpt.meta')
        saver.restore(sess, './graph.ckpt')
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        loss = tf.get_collection('loss')[0]
        global_step = tf.Variable(0, trainable=False)
        alpha = 0.1
        alpha = learning_rate_decay(alpha, 1, global_step, 10)
        train_op = tf.train.GradientDescentOptimizer(alpha).minimize(loss, global_step=global_step)
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(100):
            a = sess.run(alpha)
            print(a)
            sess.run(train_op, feed_dict={x:X, y:Y_oh})