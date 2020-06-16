#!/usr/bin/env python3

import tensorflow.keras as K
transition_layer = __import__('6-transition_layer').transition_layer

if __name__ == '__main__':
    X = K.Input(shape=(56, 56, 256))
    Y, nb_filters = transition_layer(X, 256, 0.5)
    model = K.models.Model(inputs=X, outputs=Y)
    model.summary()
    print(nb_filters)
