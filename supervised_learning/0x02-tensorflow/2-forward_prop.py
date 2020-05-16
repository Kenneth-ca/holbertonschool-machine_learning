#!/usr/bin/env python3
"""
Module to perform forward propagation
"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    a function that creates the forward propagation graph
    :param x: the placeholder for the input data
    :param layer_sizes: a list contating the number of nodes in each layer
    :param activation: a list containing the activation functions
    :return: the prediction of the network in tensor form
    """
    for i in range(len(layer_sizes)):
        if i == 0:
            prediction = create_layer(x, layer_sizes[0], activations[0])
        else:
            prediction = create_layer(prediction, layer_sizes[i],
                                      activations[i])
    return prediction
