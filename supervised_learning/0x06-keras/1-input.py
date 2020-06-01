#!/usr/bin/env python3
"""
Builds a neural network with the Keras library
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    a function that builds a NN with Keras
    :param nx: the number of input features to the network
    :param layers: a list containing the number of nodes in each layer of the
    network
    :param activations: a list containing the activation functions used for
    each layer of the network
    :param lambtha: the L2 regularization parameter
    :param keep_prob: the probability that a node will be kept for dropout
    :return: the keras model
    """
    for i in range(len(layers)):
        if i == 0:
            x = K.layers.Input(shape=(nx,))
            prev = K.layers.Dense(layers[i], activation=activations[i],
                                  kernel_regularizer=K.regularizers.l2(
                                         lambtha))
            prev = prev(x)
        else:
            reg = K.layers.Dropout(1 - keep_prob)
            reg = reg(prev)
            outputs = K.layers.Dense(layers[i], activation=activations[i],
                                     kernel_regularizer=K.regularizers.l2(
                                         lambtha))
            outputs = outputs(reg)
            prev = outputs
    model = K.Model(inputs=x, outputs=outputs)
    return model
