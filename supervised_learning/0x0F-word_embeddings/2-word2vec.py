#!/usr/bin/env python3
"""
Creates and trains a gensim word2vec model
"""
from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """
    Creates and trains a gensim word2vec model
    :param sentences: a list of sentences to be trained on
    :param size: the dimensionality of the embedding layer
    :param min_count: the minimum number of occurrences of a word for use in
    training
    :param window: the maximum distance between the current and predicted
    word within a sentence
    :param negative: the size of negative sampling
    :param cbow: boolean to determine the training type; True is for CBOW;
    False is for Skip-gram
    :param iterations: the size of negative sampling
    :param seed: the seed for the random number generator
    :param workers: the number of worker threads to train the model
    :return: the trained model
    """
    if cbow is True:
        skip = 0
    else:
        skip = 1
    model = Word2Vec(size=size, window=window,
                     min_count=min_count, workers=workers, sg=skip,
                     negative=negative, seed=seed)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=iterations)
    return model
