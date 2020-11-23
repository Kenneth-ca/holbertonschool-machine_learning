#!/usr/bin/env python3
"""
Randomly shears an image
"""
import tensorflow as tf


def shear_image(image, intensity):
    """
    Randomly shears an image
    :param image: is a 3D tf.Tensor containing the image to shear
    :param intensity: is the intensity with which the image should be sheared
    :return: the sheared image
    """
    array_img = tf.keras.preprocessing.image.img_to_array(image)
    sheared = tf.keras.preprocessing.image.random_shear(array_img, intensity,
                                                        row_axis=0, col_axis=1,
                                                        channel_axis=2)

    return tf.keras.preprocessing.image.array_to_img(sheared)
