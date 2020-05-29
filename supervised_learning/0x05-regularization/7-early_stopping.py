#!/usr/bin/env python3
"""
Determines if you should stop gradient descent early
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    a function that determines if you should stop gradient descent early
    :param cost: the current validation cost of the neural network
    :param opt_cost: the lowest recorded validation cost of the neural network
    :param threshold: the threshold used for early stopping
    :param patience: the patience count used for early stopping
    :param count: the count of how long the threshold has not been met
    :return: a boolean of whether the network should be stopped early,
    followed by the updated count
    """
    if (opt_cost - cost) > threshold:
        count = 0
    else:
        count += 1
    if count == patience:
        return True, count
    return False, count
