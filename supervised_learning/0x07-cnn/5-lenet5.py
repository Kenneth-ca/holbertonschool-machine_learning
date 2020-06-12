#!/usr/bin/env python3
"""
Builds a modified version of the LeNet-5 architecture using tensorflow
"""
import tensorflow.keras as K


def lenet5(X):
    """
    a function that builds a modified LeNet-5 using tensorflow
    :param X: tf.placeholder of shape (m, 28, 28, 1) containing the input
    images for the network
        m is the number of images
    :return:
        a tensor for the softmax activated output
        a training operation that utilizes Adam optimization (with default
        hyperparameters)
        a tensor for the loss of the network
        a tensor for the accuracy of the network
    """
    init = K.initializers.he_normal()
    activation = "relu"

    # 1st convolutional layer
    conv1 = K.layers.Conv2D(filters=6, kernel_size=(5, 5), padding="same",
                            activation=activation, kernel_initializer=init)(X)
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    # 2nd convolutional layer
    conv2 = K.layers.Conv2D(filters=16, kernel_size=(5, 5), padding="valid",
                            activation=activation, kernel_initializer=init)(
        pool1)
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    # Flatten
    flatten = K.layers.Flatten()(pool2)

    # Fully connected (FC) 1
    fc1 = K.layers.Dense(units=120, activation=activation,
                         kernel_initializer=init)(flatten)
    # FC 2
    fc2 = K.layers.Dense(units=84, activation=activation,
                         kernel_initializer=init)(fc1)
    # FC 3
    fc3 = K.layers.Dense(units=10, activation="softmax",
                         kernel_initializer=init)(fc2)

    # Model
    model = K.models.Model(inputs=X, outputs=fc3)

    opt = K.optimizers.Adam()

    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])

    return model
