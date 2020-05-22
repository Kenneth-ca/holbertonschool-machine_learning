#!/usr/bin/env python3
"""
Updates a variable using the Adam optimization algorithm
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    a function that optimizes using Adam
    :param alpha: the learning rate
    :param beta1: the weight used for the first moment
    :param beta2: the weight used for the second moment
    :param epsilon: small number to avoid division by zero
    :param var: numpy.ndarray containing the variable to be updated
    :param grad: numpy.ndarray containing the gradient of var
    :param v: the previous first moment of var
    :param s: the previous second moment of var
    :param t: the time step used for bias correction
    :return: the updated variable, the new first moment, and the new second
    moment, respectively
    """
    vdw = beta1 * v + (1 - beta1) * grad
    sdw = beta2 * s + (1 - beta2) * grad ** 2
    # Corrected
    vdw_corrected = vdw / (1 - beta1 ** t)
    sdw_corrected = sdw / (1 - beta2 ** t)
    # Updating value (var)
    var = var - alpha * (vdw_corrected / (np.sqrt(sdw_corrected) + epsilon))
    return var, vdw, sdw
