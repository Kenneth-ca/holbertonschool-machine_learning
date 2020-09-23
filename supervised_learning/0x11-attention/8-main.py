#!/usr/bin/env python3

import tensorflow as tf
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock

dblock = DecoderBlock(512, 8, 2048)
print(dblock.mha1)
print(dblock.mha2)
print(dblock.dense_hidden)
print(dblock.dense_output)
print(dblock.layernorm1)
print(dblock.layernorm2)
print(dblock.layernorm3)
print(dblock.dropout1)
print(dblock.dropout2)
print(dblock.dropout3)
x = tf.random.uniform((32, 15, 512))
hidden_states = tf.random.uniform((32, 10, 512))
output = dblock(x, hidden_states, False, None, None)
print(output)
