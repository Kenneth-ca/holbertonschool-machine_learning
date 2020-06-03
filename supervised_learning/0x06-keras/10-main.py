#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
optimize_model = __import__('2-optimize').optimize_model
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('8-train').train_model
model = __import__('9-model')
weights = __import__('10-weights')

if __name__ == '__main__':

    np.random.seed(0)
    tf.set_random_seed(0)
    network = model.load_model('network2.h5')
    weights.save_weights(network, 'weights2.h5')
    del network

    network2 = model.load_model('network1.h5')
    print(network2.get_weights())
    weights.load_weights(network2, 'weights2.h5')
    print(network2.get_weights())
