#!/usr/bin/env python3
"""
Performs the backward algorithm for a hidden markov model
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden markov model
    :param Observation: numpy.ndarray of shape (T,) that contains the index
    of the observation
        T is the number of observations
    :param Emission: numpy.ndarray of shape (N, M) containing the emission
    probability of a specific observation given a hidden state
        Emission[i, j] is the probability of observing j given the hidden
        state i
        N is the number of hidden states
        M is the number of all possible observations
    :param Transition: 2D numpy.ndarray of shape (N, N) containing the
    transition probabilities
        Transition[i, j] is the probability of transitioning from the hidden
        state i to j
    :param Initial: numpy.ndarray of shape (N, 1) containing the probability
    of starting in a particular hidden state
    :return: P, B, or None, None on failure
        Pis the likelihood of the observations given the model
        B is a numpy.ndarray of shape (N, T) containing the backward path
        probabilities
            B[i, j] is the probability of generating the future observations
            from hidden state i at time j
    """
    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
        return None, None
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None
    sum_test = np.sum(Emission, axis=1)
    if not (sum_test == 1.0).all():
        return None, None
    sum_test = np.sum(Transition, axis=1)
    if not (sum_test == 1.0).all():
        return None, None
    sum_test = np.sum(Initial, axis=0)
    if not (sum_test == 1.0).all():
        return None, None

    N, M = Emission.shape
    T = Observation.shape[0]
    if N != Transition.shape[0] or N != Transition.shape[1]:
        return None, None

    beta = np.zeros((N, T))
    beta[:, T - 1] = np.ones((N))
    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for col in range(T - 2, -1, -1):
        for row in range(N):
            beta[row, col] = np.sum(beta[:, col + 1] *
                                    Transition[row, :] *
                                    Emission[:, Observation[col + 1]])

    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * beta[:, 0])

    return P, beta
