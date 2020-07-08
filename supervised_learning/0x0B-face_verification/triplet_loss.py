#!/usr/bin/env python3
"""
Creates a class that inherits from tensorflow.keras.layers.Layer
"""
from tensorflow.keras.layers import Layer
import tensorflow.keras as K
import numpy as np


class TripletLoss(Layer):
    """
    A class that inherits from Layer
    """

    def __init__(self, alpha, **kwargs):
        """
        class constructor
        :param alpha: is the alpha value used to calculate the triplet loss
        :param kwargs: other key arguments
        """
        super(TripletLoss, self).__init__(**kwargs)
        self.alpha = alpha
        """self._dynamic = True
        self._eager_losses = True
        self._layers = True
        self._in_call = False # not working
        self._metrics = True
        self._metrics_tensors = True
        self._mixed_precision_policy = True
        self._obj_reference_counts_dict = True
        self._self_setattr_tracking = True"""

    def triplet_loss(self, inputs):
        """
        calculates the triplet loss for face recognition
        :param inputs: is a list containing the anchor, positive and negative
        output tensors from the last layer of the model, respectively
        :return: a tensor containing the triplet loss values
        """
        A, P, N = inputs

        a_p = K.layers.Subtract()([A, P])
        a_n = K.layers.Subtract()([A, N])

        a_p_2 = K.backend.sum(K.backend.square(a_p), axis=1)
        a_n_2 = K.backend.sum(K.backend.square(a_n), axis=1)

        loss = K.layers.Subtract()([a_p_2, a_n_2]) + self.alpha
        loss = K.backend.maximum(loss, 0)

        return loss

    def call(self, inputs):
        """
        calls the triplet loss and add the loss to the graph
        :param inputs: a list containing the anchor, positive, and negative
        output tensors from the last layer of the model, respectively
        :return: the triplet loss tensor
        """
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss
