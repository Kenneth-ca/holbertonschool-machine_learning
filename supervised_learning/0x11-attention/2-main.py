#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
RNNDecoder = __import__('2-rnn_decoder').RNNDecoder

decoder = RNNDecoder(2048, 128, 256, 32)
print(decoder.embedding)
print(decoder.gru)
print(decoder.F)
x = tf.convert_to_tensor(np.random.choice(2048, 32).reshape((32, 1)))
s_prev = tf.convert_to_tensor(np.random.uniform(size=(32, 256)).astype('float32'))
hidden_states = tf.convert_to_tensor(np.random.uniform(size=(32, 10, 256)).astype('float32'))
y, s = decoder(x, s_prev, hidden_states)
print(y)
print(s)
