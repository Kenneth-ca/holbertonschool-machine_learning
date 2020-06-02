#!/usr/bin/env python3
"""
That trains a model using mini-batch gradient descent
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, verbose=True,
                shuffle=False):
    """
    a function that trains a model using mini-batch gradient descent
    :param network: the model to train
    :param data: numpy.ndarray of shape (m, nx) containing the input data
    :param labels: a one-hot numpy.ndarray of shape (m, classes) containing
    the labels of data
    :param batch_size: the size of the batch used for mini-batch gradient
    descent
    :param epochs: the number of passes through data for mini-batch gradient
    descent
    :param verbose: a boolean that determines if output should be printed
    during training
    :param shuffle: a boolean that determines whether to shuffle the batches
    every epoch. Normally, it is a good idea to shuffle, but for
    reproducibility, we have chosen to set the default to False
    :return: the History object generated after training the model
    """
    history = network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                          verbose=verbose, shuffle=shuffle)
    return history
