#!/usr/bin/env python3

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block

if __name__ == '__main__':
    X = K.Input(shape=(56, 56, 64))
    Y, nb_filters = dense_block(X, 64, 32, 6)
    model = K.models.Model(inputs=X, outputs=Y)
    model.summary()
    print(nb_filters)
