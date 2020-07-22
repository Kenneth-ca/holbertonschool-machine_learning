#!/usr/bin/env python3
"""
Creates the class TrainModel
"""
from triplet_loss import TripletLoss
import tensorflow.keras as K
import tensorflow as tf
import numpy as np


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
        trains self.training_model
        :param triplets: a list of numpy.ndarrayscontaining the inputs to
        self.training_model
        :param epochs: the number of epochs to train for
        :param batch_size: is the batch size for training
        :param validation_split: is the validation split for training
        :param verbose: is a boolean that sets the verbosity mode
        :return: the History output from the training
        """
        history = self.training_model.fit(triplets,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=verbose,
                                          validation_split=validation_split)
        return history

    def save(self, save_path):
        """
        saves the base embedding model
        :param save_path:
        :return:
        """
        K.models.save_model(self.base_model, save_path)
        return self.base_model

    @staticmethod
    def f1_score(y_true, y_pred):
        """
        calculates the F1 score of predictions
        :param y_true: numpy.ndarray of shape (m,) containing the correct labels
            m is the number of examples
        :param y_pred: numpy.ndarray of shape (m,) containing the predicted
        labels
        :return: The f1 score
        """
        TP = np.count_nonzero(y_pred * y_true)
        FP = np.count_nonzero(y_pred * (y_true - 1))
        FN = np.count_nonzero((y_pred - 1) * y_true)

        div0 = TP + FP
        div1 = TP + FN

        if div0 == 0 or div1 == 0:
            return 0

        precision = TP / div0
        recall = TP / div1
        f1 = 2 * precision * recall / (precision + recall)

        return f1

    @staticmethod
    def accuracy(y_true, y_pred):
        """

        :param y_true:
        :param y_pred:
        :return:
        """
        TP = np.count_nonzero(y_pred * y_true)
        TN = np.count_nonzero((y_pred - 1) * (y_true - 1))
        FP = np.count_nonzero(y_pred * (y_true - 1))
        FN = np.count_nonzero((y_pred - 1) * y_true)

        accuracy = (TP + TN) / (TP + FN + TN + FP)

        return accuracy

    def best_tau(self, images, identities, thresholds):
        """
        calculates the best tau to use for a maximal F1 score
        :param images: numpy.ndarray of shape (m, n, n, 3) containing the
        aligned images for testing
            m is the number of images
            n is the size of the images
        :param identities: list containing the identities of each image in
        images
        :param thresholds: a 1D numpy.ndarray of distance thresholds (tau) to
        test
        :return: (tau, f1, acc)
            tau- the optimal threshold to maximize F1 score
            f1 - the maximal F1 score
            acc - the accuracy associated with the maximal F1 score
        """
        def distance(emb1, emb2):
            return np.sum(np.square(emb1 - emb2))

        embedded = np.zeros((images.shape[0], 128))

        for i, m in enumerate(images):
            embedded[i] = self.base_model.predict(np.expand_dims(m,
                                                                 axis=0))[0]

        distances = []
        identical = []

        num = len(identities)

        for i in range(num - 1):
            for j in range(i + 1, num):
                distances.append(distance(embedded[i], embedded[j]))
                identical.append(
                    1 if identities[i] == identities[j] else 0)

        distances = np.array(distances)
        identical = np.array(identities)

        f1_scores = [self.f1_score(identical, distances < t) for t in
                     thresholds]
        acc_scores = [self.accuracy(identical, distances < t) for t in
                      thresholds]

        opt_idx = np.argmax(f1_scores)

        opt_tau = thresholds[opt_idx]
        opt_f1 = f1_scores[opt_idx]
        opt_acc = acc_scores[opt_idx]

        return opt_tau, opt_f1, opt_acc
