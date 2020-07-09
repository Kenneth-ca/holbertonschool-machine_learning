#!/usr/bin/env python3
"""
Creates the class TrainModel
"""
from triplet_loss import TripletLoss
import tensorflow.keras as K
import tensorflow as tf


class TrainModel:
    """
    A class that trains a model for face verification
    """

    def __init__(self, model_path, alpha):
        """
        class constructor
        :param model_path: is the path to the base face verification
        embedding model
        :param alpha: is the alpha to use for the triplet loss calculation
        """
        with tf.keras.utils.CustomObjectScope({'tf': tf}):
            self.base_model = K.models.load_model(model_path)

        A = K.Input(shape=(96, 96, 3))
        P = K.Input(shape=(96, 96, 3))
        N = K.Input(shape=(96, 96, 3))

        inputs = [A, P, N]

        output_A = self.base_model(A)
        output_P = self.base_model(P)
        output_N = self.base_model(N)

        output0 = [output_A, output_P, output_N]
        loss = TripletLoss(alpha)
        outputs = loss(output0)

        model = K.models.Model(inputs, outputs)
        model.compile(optimizer="Adam")

        self.training_model = model

    def train(self, triplets, epochs=5, batch_size=32, validation_split=0.3,
              verbose=True):
        """

        :param triplets:
        :param epochs:
        :param batch_size:
        :param validation_split:
        :param verbose:
        :return:
        """
        history = self.training_model.fit(triplets,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=verbose,
                                          validation_split=validation_split)
        return history

    def save(self, save_path):
        """

        :param save_path:
        :return:
        """
