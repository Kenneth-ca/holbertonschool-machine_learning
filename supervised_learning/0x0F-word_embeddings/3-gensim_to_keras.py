#!/usr/bin/env python3
"""
Converts a gensim word2vec model to a keras Embedding layer
"""
from gensim.models import Word2Vec
import tensorflow.keras as keras


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a keras Embedding layer
    :param model:
    :return:
    """
    return model.wv.get_keras_embedding(train_embeddings=False)
