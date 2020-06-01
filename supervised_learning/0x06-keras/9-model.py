#!/usr/bin/env python3
"""
Saves and load Model for Keras
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    saves an entire model
    :param network: the model to save
    :param filename: is the path of the file that the model should be saved to
    :return: None
    """
    K.models.save_model(model=network, filepath=filename)
    return None


def load_model(filename):
    """
    loads an entire model
    :param filename: the path of the file that the model should be loaded from
    :return: the loaded model
    """
    return K.models.load_model(filepath=filename)
