#!/usr/bin/env python3
"""
Updates a variable using the gradient descent with momentum with tensorflow
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    a function that uses momentum optimization algorithm with tensorflow
    :param loss: the loss of the network
    :param alpha: the learning rate
    :param beta1: the momentum weight
    :return: the momentum optimization
    """
    # the minimize method of MomentumOptimizer simply the compute_gradients()
    # and apply_gradients() calls
    return tf.train.MomentumOptimizer(alpha, momentum=beta1).minimize(loss)
