#!/usr/bin/env python3
"""
Has the trained agent play an episode
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode
    :param env: is the FrozenLakeEnv instance
    :param Q: is a numpy.ndarray containing the Q-table
    :param max_steps: is the maximum number of steps in the episode
    :return: the total rewards for the episode
    """
    state = env.reset()
    env.render()
    for step in range(max_steps):
        action = np.argmax(Q[state])
        new_state, reward, done, info = env.step(action)
        env.render()
        if done:
            return reward
        state = new_state

    env.close()
