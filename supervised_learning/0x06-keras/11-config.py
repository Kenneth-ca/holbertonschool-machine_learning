#!/usr/bin/env python3
"""
Saves and load Model for Keras
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    saves an entire model
    :param network: the model to save
    :param filename: is the path of the file that the model should be saved to
    :return: None
    """
    # serialize model to json
    json_network = network.to_json()
    with open(filename, 'w') as f:
        f.write(json_network)
    return None


def load_config(filename):
    """
    loads an entire model
    :param filename: the path of the file containing the model’s
    configuration in JSON format
    :return: the loaded model
    """
    with open(filename, 'r') as f:
        json_saved = f.read()
    return K.models.model_from_json(json_saved)
