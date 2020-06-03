#!/usr/bin/env python3
"""Updates train model with early stopping"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """
    a function that trains the model using early stopping
    :param network: is the model to train
    :param data: ndarray of shape (m, nx) containing the input data
    :param labels: is a one-hot numpy.ndarray of shape (m, classes)
    containing the labels of data
    :param batch_size: size of the batch used for mini-batch gradient descent
    :param epochs:number of passes through data for mini-batch gradient descent
    :param validation_data: is the data to validate the model with, if not None
    :param early_stopping: indicates whether early stopping should be used
    early stopping should only be performed if validation_data exists
    early stopping should be based on validation loss
    :param patience: is the patience used for early stopping
    :param verbose: boolean that determines if output should be printed
    :param shuffle: boolean that determines whether to shuffle the batches
    every epoch. Normally, it is a good idea to shuffle, but for
    reproducibility, we have chosen to set the default to False.
    :return:
    """
    stopping = []
    if early_stopping and validation_data:
        stopping.append(K.callbacks.EarlyStopping(patience=patience))
    history = network.fit(data, labels, epochs=epochs, batch_size=batch_size,
                          shuffle=shuffle, verbose=verbose, callbacks=stopping,
                          validation_data=validation_data)
    return history
