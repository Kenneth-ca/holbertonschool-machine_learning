#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from triplet_loss import TripletLoss

np.random.seed(0)
tl = TripletLoss(0.2)
A = np.random.uniform(0, 1, (2, 128))
P = np.random.uniform(0, 1, (2, 128))
N = np.random.uniform(0, 1, (2, 128))

with tf.Session() as sess:
    loss = tl.triplet_loss([A, P, N])
    print(type(loss))
    print(loss.eval())
