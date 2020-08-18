#!/usr/bin/env python3

BO = __import__('4-bayes_opt').BayesianOptimization
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    """our 'black box' function"""
    return np.sin(5*x) + 2*np.sin(-2*x)

if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    bo = BO(f, X_init, Y_init, (-np.pi, 2*np.pi), 50, l=0.6, sigma_f=2, xsi=0.05)
    X_next, EI = bo.acquisition()

    print(EI)
    print(X_next)

    plt.scatter(X_init.reshape(-1), Y_init.reshape(-1), color='g')
    plt.plot(bo.X_s.reshape(-1), EI.reshape(-1), color='r')
    plt.axvline(x=X_next)
    plt.show()
