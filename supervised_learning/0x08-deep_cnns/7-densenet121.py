#!/usr/bin/env python3
"""
Builds a DenseNet-121 architecture
"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    a function that builds a DenseNet-121 architecture
    :param growth_rate: is the growth rate for the dense block
    :param compression: is the compression factor for the transition layer
    :return: The keras model
    """
    init = K.initializers.he_normal()
    X = K.Input(shape=(224, 224, 3))
    n_filters = 2 * growth_rate

    norm1 = K.layers.BatchNormalization()(X)
    act1 = K.layers.Activation("relu")(norm1)
    conv1 = K.layers.Conv2D(filters=n_filters, kernel_size=(7, 7),
                            padding="same",
                            strides=(2, 2),
                            kernel_initializer=init)(act1)

    max_pool = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                     padding="same")(conv1)

    # First block
    dense_block1, n_filters = dense_block(max_pool, n_filters, growth_rate, 6)

    transition_layer1, n_filters = transition_layer(dense_block1, n_filters,
                                                    compression)

    # Second block
    dense_block2, n_filters = dense_block(transition_layer1, n_filters,
                                          growth_rate, 12)

    transition_layer2, n_filters = transition_layer(dense_block2, n_filters,
                                                    compression)

    # Third Block
    dense_block3, n_filters = dense_block(transition_layer2, n_filters,
                                          growth_rate, 24)

    transition_layer3, n_filters = transition_layer(dense_block3, n_filters,
                                                    compression)

    # Last block (no transition)
    dense_block4, n_filters = dense_block(transition_layer3, n_filters,
                                          growth_rate, 16)

    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1),
                                         padding="valid")(dense_block4)

    softmax = K.layers.Dense(units=1000, activation="softmax",
                             kernel_initializer=init)(avg_pool)

    model = K.Model(inputs=X, outputs=softmax)

    return model
