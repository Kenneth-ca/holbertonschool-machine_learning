#!/usr/bin/env python3
"""
Tests a Neural Network
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    a function that tests a NN
    :param network: the network model to test
    :param data: the input data to test the model with
    :param labels: the correct one-hot labels of data
    :param verbose: a boolean that determines if output should be printed
    during the testing process
    :return: the loss and accuracy of the model with the testing data,
    respectively
    """
    return network.evaluate(data, labels, verbose=verbose)
