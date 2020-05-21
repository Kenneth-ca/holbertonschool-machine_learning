#!/usr/bin/env python3
"""
Trains a loaded neural network model using mini-batch gradient descent
"""
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    a function that trains a loaded neural network model using mini-batch
    gradient descent
    :param X_train: np.ndarray of shape (m, 784) containing the training data
        m is the number of data points
        784 is the number of input features
    :param Y_train: one-hot numpy.ndarray of shape (m, 10) containing the
    training labels
        10 is the number of classes the model should classify
    :param X_valid: np.ndarray of shape (m, 784) containing the validation data
    :param Y_valid: one-hot np.ndarray of shape (m, 10) containing the
    validation labels
    :param batch_size: number of data points in a batch
    :param epochs: number of times the training should pass through the whole
    dataset
    :param load_path: path from which to load the model
    :param save_path: path to where the model should be saved after training
    :return: path where the model was saved
    """
