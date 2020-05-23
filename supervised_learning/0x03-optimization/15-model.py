#!/usr/bin/env python3
"""
Optimizes a neural network with tensorflow
"""
import tensorflow as tf


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32,
          epochs=5, save_path='/tmp/model.ckpt'):
    """
    a function that optimizes a neural network model with tensorflow
    :param Data_train: tuple containing the training inputs and training
    labels, respectively
    :param Data_valid: tuple containing the validation inputs and validation
    labels, respectively
    :param layers: a list containing the number of nodes in each layer of the networ
    :param activations: a list containing the activation functions used for
    each layer of the network
    :param alpha: the learning rate
    :param beta1: the weight for the first moment of Adam Optimization
    :param beta2: the weight for the second moment of Adam Optimization
    :param epsilon: a small number used to avoid division by zero
    :param decay_rate: the decay rate for inverse time decay of the learning
    rate (the corresponding decay step should be 1)
    :param batch_size: the number of data points that should be in a mini-batch
    :param epochs: the number of times the training should pass through the
    whole dataset
    :param save_path: the path where the model should be saved to
    :return: the path where the model was saved
    """
