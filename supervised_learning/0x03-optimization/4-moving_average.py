#!/usr/bin/env python3
"""
Calculates the weighted moving average of a data set:
"""


def moving_average(data, beta):
    """
    a function that calculates the weighted moving average of a data set
    :param data: the list of data to calculate the moving average of
    :param beta: the weight used for the moving average
    :return: a list containing the moving averages of data
    """
    if beta > 1 or beta < 0:
        return None
    vt = 0
    moving = []
    for i in range(len(data)):
        # Moving average: Vt = beta * Vt-1 + (1 - beta) * data[i]
        vt = beta * vt + (1 - beta) * data[i]
        # Correction of bias: vt / (1 - beta)
        correction = 1 - beta ** (i + 1)
        moving.append(vt / correction)
    return moving
