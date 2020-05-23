#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
create_batch_norm_layer = __import__('14-batch_norm').create_batch_norm_layer

if __name__ == '__main__':
    lib= np.load('../data/MNIST.npz')
    X_3D = lib['X_train']
    X = X_3D.reshape((X_3D.shape[0], -1))

    tf.set_random_seed(0)
    x = tf.placeholder(tf.float32, shape=[None, 784])
    a = create_batch_norm_layer(x, 256, tf.nn.tanh)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(a, feed_dict={x:X[:5]}))