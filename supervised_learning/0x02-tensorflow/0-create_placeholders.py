#!/usr/bin/env python3
"""
Module to create a placeholder
"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    a function that create placeholders
    :param nx: the number of feature columns in our data
    :param classes: the number of classes in our classifier
    :return: placeholders named x and y, respectively
    """
    return tf.placeholder(float, shape=[None, nx], name='x'), tf.placeholder(
        float, shape=[None, classes], name='y')
