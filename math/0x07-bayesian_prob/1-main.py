#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    intersection = __import__('1-intersection').intersection

    P = np.linspace(0, 1, 11)
    Pr = np.ones(11) / 11 # this prior assumes that everything is equally as likely
    print(intersection(26, 130, P, Pr))
