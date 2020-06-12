#!/usr/bin/env python3
"""
Builds a modified version of the LeNet-5 architecture using tensorflow
"""
import tensorflow as tf


def lenet5(x, y):
    """
    a function that builds a modified LeNet-5 using tensorflow
    :param x: tf.placeholder of shape (m, 28, 28, 1) containing the input
    images for the network
        m is the number of images
    :param y: tf.placeholder of shape (m, 10) containing the one-hot labels
    for the network
    :return:
        a tensor for the softmax activated output
        a training operation that utilizes Adam optimization (with default
        hyperparameters)
        a tensor for the loss of the network
        a tensor for the accuracy of the network
    """
    init = tf.contrib.layers.variance_scaling_initializer()
    activation = tf.nn.relu

    # 1st convolutional layer
    conv1 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5), padding="same",
                             activation=activation, kernel_initializer=init)(x)
    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    # 2nd convolutional layer
    conv2 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5), padding="valid",
                             activation=activation, kernel_initializer=init)(
        pool1)
    pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    # Flatten
    flatten = tf.layers.Flatten()(pool2)

    # Fully connected (FC) 1
    fc1 = tf.layers.Dense(units=120, activation=activation,
                          kernel_initializer=init)(flatten)
    # FC 2
    fc2 = tf.layers.Dense(units=84, activation=activation,
                          kernel_initializer=init)(fc1)
    # FC 3
    fc3 = tf.layers.Dense(units=10, activation=None,
                          kernel_initializer=init)(fc2)

    # Prediction
    y_pred = tf.nn.softmax(fc3)

    # Loss
    loss = tf.losses.softmax_cross_entropy(y, fc3)

    # Accuracy
    accuracy = tf.equal(tf.argmax(y, 1), tf.argmax(fc3, 1))
    mean = tf.reduce_mean(tf.cast(accuracy, tf.float32))

    # Train
    train = tf.train.AdamOptimizer().minimize(loss)

    return y_pred, train, loss, mean
