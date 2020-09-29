#!/usr/bin/env python3
"""
Performs Q-learning
"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning
    :param env: is the FrozenLakeEnv instance
    :param Q: is a numpy.ndarray containing the Q-table
    :param episodes: is the total number of episodes to train over
    :param max_steps: is the maximum number of steps per episode
    :param alpha: is the learning rate
    :param gamma: is the discount rate
    :param epsilon: is the initial threshold for epsilon greedy
    :param min_epsilon: is the minimum value that epsilon should decay to
    :param epsilon_decay: is the decay rate for updating epsilon between
    episodes
    :return: Q, total_rewards
        Q is the updated Q-table
        total_rewards is a list containing the rewards per episode
    """
    total_rewards_list = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_rewards = 0
        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, info = env.step(action)

            Q[state, action] = (Q[state, action] +
                                alpha * (reward +
                                         gamma * np.max(Q[new_state, :]) - Q[
                                             state, action]))
            state = new_state

            if done is True:
                if reward == 0.0:
                    total_rewards = -1
                total_rewards += reward
                break
            total_rewards += reward

        epsilon = min_epsilon + (1 - min_epsilon) * np.exp(
                -epsilon_decay * episode)
        total_rewards_list.append(total_rewards)
    return Q, total_rewards_list
