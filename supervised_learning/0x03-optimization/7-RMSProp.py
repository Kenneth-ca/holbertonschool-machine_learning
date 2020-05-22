#!/usr/bin/env python3
"""
Updates a variable using the RMSprop optimization algorithm
"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    a function that updates a variable using RMSprop
    :param alpha: the learning rate
    :param beta2: the RMSprop weight
    :param epsilon: a small number to avoid division by zero
    :param var: numpy.ndarray containing the variable to be updated
    :param grad: numpy.ndarray containing the gradient of var
    :param s: is the previous second moment of var
    :return: the updated variable and the new moment, respectively
    """
    new_moment = beta2 * s + (1 - beta2) * grad ** 2
    updated = var - alpha * grad / (new_moment ** (1 / 2) + epsilon)
    return updated, new_moment
