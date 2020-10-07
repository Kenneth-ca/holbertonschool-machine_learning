#!/usr/bin/env python3
import tensorflow as tf
train_transformer = __import__('5-train').train_transformer

tf.compat.v1.enable_eager_execution()
transformer = train_transformer(6, 512, 8, 2048, 32, 40, 5)
