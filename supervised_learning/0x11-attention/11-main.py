#!/usr/bin/env python3

import tensorflow as tf
Transformer = __import__('11-transformer').Transformer

transformer = Transformer(6, 512, 8, 2048, 10000, 12000, 1000, 1500)
print(transformer.encoder)
print(transformer.decoder)
print(transformer.linear)
x = tf.random.uniform((32, 10))
y = tf.random.uniform((32, 15))
output = transformer(x, y, True, None, None, None)
print(output)
