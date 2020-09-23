#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention

np.random.seed(0)
Q = tf.convert_to_tensor(np.random.uniform(size=(50, 10, 256)).astype('float32'))
K = tf.convert_to_tensor(np.random.uniform(size=(50, 15, 256)).astype('float32'))
V = tf.convert_to_tensor(np.random.uniform(size=(50, 15, 512)).astype('float32'))
output, weights = sdp_attention(Q, K, V)
print(output)
print(weights)
