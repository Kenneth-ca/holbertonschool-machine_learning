#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from triplet_loss import TripletLoss

np.random.seed(0)
tl = TripletLoss(0.2)
A = np.random.uniform(0, 1, (2, 128))
P = np.random.uniform(0, 1, (2, 128))
N = np.random.uniform(0, 1, (2, 128))

inputs = [tf.keras.Input((128,)), tf.keras.Input((128,)), tf.keras.Input((128,))]
output = tl(inputs)

model = tf.keras.models.Model(inputs, output)
print(output)
print(tl._losses)
print(model.losses)
print(model.predict([A, P, N]))
