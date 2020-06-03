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
    K.Model()
    inputs = K.Input(shape=(nx,))
    x = K.layers.Dense(layers[0], activation=activations[0],
                       kernel_regularizer=K.regularizers.l2(lambtha))(inputs)
    y = x
    rate = 1 - keep_prob
    for i in range(1, len(layers)):
        if i == 1:
            y = K.layers.Dropout(rate)(x)
        else:
            y = K.layers.Dropout(rate)(y)
        y = K.layers.Dense(layers[i], activation=activations[i],
                           kernel_regularizer=K.regularizers.l2(lambtha))(y)
    model = K.Model(inputs=inputs, outputs=y)
    return model
