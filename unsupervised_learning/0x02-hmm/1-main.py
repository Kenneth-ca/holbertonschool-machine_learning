#!/usr/bin/env python3

import numpy as np
regular = __import__('1-regular').regular

if __name__ == '__main__':
    a = np.eye(2)
    b = np.array([[0.6, 0.4],
                  [0.3, 0.7]])
    c = np.array([[0.25, 0.2, 0.25, 0.3],
                  [0.2, 0.3, 0.2, 0.3],
                  [0.25, 0.25, 0.4, 0.1],
                  [0.3, 0.3, 0.1, 0.3]])
    d = np.array([[0.8, 0.2, 0, 0, 0],
                [0.25, 0.75, 0, 0, 0],
                [0, 0, 0.5, 0.2, 0.3],
                [0, 0, 0.3, 0.5, .2],
                [0, 0, 0.2, 0.3, 0.5]])
    e = np.array([[1, 0.25, 0, 0, 0],
                [0.25, 0.75, 0, 0, 0],
                [0, 0.1, 0.5, 0.2, 0.2],
                [0, 0.1, 0.2, 0.5, .2],
                [0, 0.1, 0.2, 0.2, 0.5]])
    print(regular(a))
    print(regular(b))
    print(regular(c))
    print(regular(d))
    print(regular(e))
