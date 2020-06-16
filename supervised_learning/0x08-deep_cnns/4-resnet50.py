#!/usr/bin/env python3
"""
Builds a ResNet-50 architecture
"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    a function that builds a ResNet-50 architecture
    :return: the keras model
    """
    init = K.initializers.he_normal()
    X = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2),
                            padding="same", kernel_initializer=init)(X)
    norm1 = K.layers.BatchNormalization()(conv1)
    act1 = K.layers.Activation("relu")(norm1)

    max_pool = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                     padding="same")(act1)

    # Conv2_x blocks
    conv2_x_1 = projection_block(max_pool, [64, 64, 256], 1)

    conv2_x_2 = identity_block(conv2_x_1, [64, 64, 256])
    conv2_x_3 = identity_block(conv2_x_2, [64, 64, 256])

    # Conv3_x blocks
    conv3_x_1 = projection_block(conv2_x_3, [128, 128, 512])

    conv3_x_2 = identity_block(conv3_x_1, [128, 128, 512])
    conv3_x_3 = identity_block(conv3_x_2, [128, 128, 512])
    conv3_x_4 = identity_block(conv3_x_3, [128, 128, 512])

    # Conv4_x blocks

    conv4_x_1 = projection_block(conv3_x_4, [256, 256, 1024])

    conv4_x_2 = identity_block(conv4_x_1, [256, 256, 1024])
    conv4_x_3 = identity_block(conv4_x_2, [256, 256, 1024])
    conv4_x_4 = identity_block(conv4_x_3, [256, 256, 1024])
    conv4_x_5 = identity_block(conv4_x_4, [256, 256, 1024])
    conv4_x_6 = identity_block(conv4_x_5, [256, 256, 1024])

    # Conv5_x blocks
    conv5_x_1 = projection_block(conv4_x_6, [512, 512, 2048])

    conv5_x_2 = identity_block(conv5_x_1, [512, 512, 2048])
    conv5_x_3 = identity_block(conv5_x_2, [512, 512, 2048])

    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1),
                                         padding="valid")(conv5_x_3)

    softmax = K.layers.Dense(units=1000, activation="softmax",
                             kernel_initializer=init)(avg_pool)

    model = K.Model(inputs=X, outputs=softmax)

    return model
