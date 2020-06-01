#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
optimize_model = __import__('2-optimize').optimize_model
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('8-train').train_model
model = __import__('9-model')
config = __import__('11-config')

if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)
    network = model.load_model('network1.h5')
    config.save_config(network, 'config1.json')
    del network

    network2 = config.load_config('config1.json')
    network2.summary()
    print(network2.get_weights())