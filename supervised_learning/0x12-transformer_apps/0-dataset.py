#!/usr/bin/env python3
"""
Create the class Dataset that loads and preps a dataset for machine translation
"""
import tensorflow_datasets as tfds
import tensorflow as tf


class Dataset:
    """
    Loads and preps a dataset
    """

    def __init__(self):
        """
        Class constructor
        """
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)
        self.data_train = examples['train']
        self.data_valid = examples['validation']

        self.tokenizer_en = self.tokenize_dataset(self.data_train)

        self.tokenizer_pt = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for our dataset
        :param data: a tf.data.Dataset whose examples are formatted as a
        tuple (pt, en)
            pt is the tf.Tensor containing the Portuguese sentence
            en is the tf.Tensor containing the corresponding English sentence
        :return: tokenizer_pt, tokenizer_en
            tokenizer_pt is the Portuguese tokenizer
            tokenizer_en is the English tokenizer
        """
        tokenizer_pt = (tfds.features.
                        text.SubwordTextEncoder.
                        build_from_corpus((en.
                                          numpy() for pt,en in self.data_train),
                                          target_vocab_size=2 ** 15))
        tokenizer_en = (tfds.features.
                        text.SubwordTextEncoder.
                        build_from_corpus((pt.
                                          numpy() for pt,en in self.data_train),
                                          target_vocab_size=2 ** 15))

        return tokenizer_pt, tokenizer_en