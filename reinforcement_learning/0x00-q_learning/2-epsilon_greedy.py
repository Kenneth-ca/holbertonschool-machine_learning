#!/usr/bin/env python3
"""
Uses epsilon-greedy to determine the next action
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action
    :param Q: numpy.ndarray containing the q-table
    :param state: the current state
    :param epsilon: is the epsilon to use for the calculation
    :return: the next action index
    """
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.randint(0, Q.shape[1])
    else:
        action = np.argmax(Q[state])
    return action
