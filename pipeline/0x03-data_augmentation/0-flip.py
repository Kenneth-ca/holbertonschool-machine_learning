#!/usr/bin/env python3
"""
Flips an image horizontally
"""
import tensorflow as tf


def flip_image(image):
    """
    Flips an image horizontally
    :param image: is a 3D tf.Tensor containing the image to flip
    :return: the flipped image
    """
    return tf.image.flip_left_right(image)
