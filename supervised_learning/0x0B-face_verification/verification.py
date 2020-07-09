#!/usr/bin/env python3
"""
Creates the class FaceVerification
"""
import numpy as np


class FaceVerification:
    """
    a class to perform face verification
    """

    def __init__(self, model_path, database, identities):
        """

        :param model_path: the path to where the face verification embedding
        model is stored
        :param database: a numpy.ndarray of shape (d, e) containing all the
        face embeddings in the database
            d is the number of images in the database
            e is the dimensionality of the embedding
        :param identities: a list of length d containing the identities
        corresponding to the embeddings in database
        """
        with tf.keras.utils.CustomObjectScope({'tf': tf}):
            self.model = K.models.load_model(model_path)
        self.database = database
        self.identities = identities

    def embedding(self, images):
        """
        calculates the face embedding of images
        :param images: numpy.ndarray of shape (i, n, n, 3) containing the
        aligned images
            i is the number of images
            n is the size of the aligned images
        :return: a numpy.ndarray of shape (i, e) containing the embeddings
        where e is the dimensionality of the embeddings
        """
        embeddings = np.zeros((images.shape[0], 128))
        for i, m in enumerate(images):
            embeddings[i] = self.model.predict(np.expand_dims(m, axis=0))[0]
        return embeddings
