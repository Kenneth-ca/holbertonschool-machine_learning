#!/usr/bin/env python3

import numpy as np
l2_reg_cost = __import__('0-l2_reg_cost').l2_reg_cost

if __name__ == '__main__':
    np.random.seed(0)

    weights = {}
    weights['W1'] = np.random.randn(256, 784)
    weights['W2'] = np.random.randn(128, 256)
    weights['W3'] = np.random.randn(10, 128)

    cost = np.abs(np.random.randn(1))

    print(cost)
    cost = l2_reg_cost(cost, 0.1, weights, 3, 1000)
    print(cost)