#!/usr/bin/env python3
"""
Updates the funtion train model to analyze validation data
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """
    a function that trains a model using mini-batch gradient descent
    :param network: the model to train
    :param data: numpy.ndarray of shape (m, nx) containing the input data
    :param labels: a one-hot numpy.ndarray of shape (m, classes) containing
    the labels of data
    :param batch_size: the size of the batch used for mini-batch gradient
    descent
    :param epochs: the number of passes through data for mini-batch gradient
    descent
    :param validation_data: data to validate the model with
    :param early_stopping: indicates whether early stopping should be used
    :param patience: the patience used for early stopping
    :param learning_rate_decay: indicates if the decay is going to be performed
    :param alpha: the initial learning rate
    :param decay_rate: the decay rate
    :param save_best: indicates if the model will be saved when it is the best
    :param filepath: the file path where the model should be saved
    :param verbose: a boolean that determines if output should be printed
    during training
    :param shuffle: a boolean that determines whether to shuffle the batches
    every epoch. Normally, it is a good idea to shuffle, but for
    reproducibility, we have chosen to set the default to False
    :return: the History object generated after training the model
    """

    def l_r_decay(epoch):
        """
        a function that performs step decay
        :param epoch:
        :return:
        """
        return alpha / (1 + decay_rate * epoch)

    callbacks = []
    if early_stopping and validation_data:
        callbacks.append(K.callbacks.EarlyStopping(patience=patience,
                                                   monitor="val_loss"))
    if learning_rate_decay and validation_data:
        callbacks.append(K.callbacks.LearningRateScheduler(l_r_decay,
                                                           verbose=1))
    if save_best and validation_data:
        callbacks.append(K.callbacks.ModelCheckpoint(filepath=filepath,
                                                     save_best_only=True))
    history = network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                          verbose=verbose, validation_data=validation_data,
                          shuffle=shuffle, callbacks=callbacks)
    return history
