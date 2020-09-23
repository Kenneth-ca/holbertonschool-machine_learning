#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock

eblock = EncoderBlock(512, 8, 2048)
print(eblock.mha)
print(eblock.dense_hidden)
print(eblock.dense_output)
print(eblock.layernorm1)
print(eblock.layernorm2)
print(eblock.dropout1)
print(eblock.dropout2)
x = tf.random.uniform((32, 10, 512))
output = eblock(x, True, None)
print(output)
