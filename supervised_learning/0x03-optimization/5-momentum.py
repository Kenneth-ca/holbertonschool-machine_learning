#!/usr/bin/env python3
"""
Updates a variable using the gradient descent with momentum
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    a function that uses momentum optimization algorithm
    :param alpha: the learning rate
    :param beta1: the momentum weight
    :param var: numpy.ndarray containing the variable to be updated
    :param grad: numpy.ndarray containing the gradient of var
    :param v: the previous first moment of var
    :return: the updated variable and the new moment, respectively
    """
    momentum = beta1 * v + (1 - beta1) * grad
    updated = var - alpha * momentum
    return updated, momentum
