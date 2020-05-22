#!/usr/bin/env python3
"""
Updates the learning rate using inverse time decay in numpy
"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    a function that updates the learning rate
    :param alpha: the original learning rate
    :param decay_rate: the weight used to determine the rate at which alpha
    will decay
    :param global_step: the number of passes of gradient descent that have
    elapsed
    :param decay_step: the number of passes of gradient descent that should
    occur before alpha is decayed further
    :return: the updated value for alpha
    """
    epoch = global_step // decay_step
    new_alpha = alpha / (1 + decay_rate * epoch)
    return new_alpha
