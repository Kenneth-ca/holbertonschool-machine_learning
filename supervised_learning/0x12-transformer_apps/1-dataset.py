#!/usr/bin/env python3
"""
Create the class Dataset that loads and preps a dataset for machine translation
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


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

        en, pt = self.tokenize_dataset(self.data_train)

        self.tokenizer_en = en
        self.tokenizer_pt = pt

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
                        build_from_corpus((en.numpy() for pt,
                                          en in self.data_train),
                                          target_vocab_size=2 ** 15))
        tokenizer_en = (tfds.features.
                        text.SubwordTextEncoder.
                        build_from_corpus((pt.numpy() for pt,
                                          en in self.data_train),
                                          target_vocab_size=2 ** 15))

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens
        :param pt: the tf.Tensor containing the Portuguese sentence
        :param en: the tf.Tensor containing the corresponding English sentence
        :return: pt_tokens, en_tokens
            pt_tokens is a tf.Tensor containing the Portuguese tokens
            en_tokens is a tf.Tensor containing the English tokens
        """
        lang1 = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]

        lang2 = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size + 1]

        return lang1, lang2
