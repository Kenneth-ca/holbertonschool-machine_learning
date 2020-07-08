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

        """loss = TripletLoss(alpha)

        A_input = tf.placeholder(tf.float32, (None, 96, 96, 3))
        P_input = tf.placeholder(tf.float32, (None, 96, 96, 3))
        N_input = tf.placeholder(tf.float32, (None, 96, 96, 3))
        inputs = [A_inputs, P_inputs, N_inputs]

        outputs = self.base_model(inputs)"""