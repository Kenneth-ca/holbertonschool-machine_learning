#!/usr/bin/env python3
"""
Creates the class RNNDecoder
Helpful article:
https://towardsdatascience.com/implementing-neural-machine-translation-with
-attention-using-tensorflow-fc9c6f26155f
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    Class to decode for machine translation:
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor
        :param vocab: an integer representing the size of the input vocabulary
        :param embedding: an integer representing the dimensionality of the
        embedding vector
        :param units: an integer representing the number of hidden units in
        the RNN cell
        :param batch: an integer representing the batch size
        """
        super(RNNDecoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer="glorot_uniform",
                                       return_sequences=True,
                                       return_state=True)
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(self.units)

    def call(self, x, s_prev, hidden_states):
        """
        This function performs the embedding
        :param x: x is a tensor of shape (batch, 1) containing the previous
        word in the target sequence as an index of the target vocabulary
        :param s_prev: a tensor of shape (batch, units) containing the
        previous decoder hidden state
        :param hidden_states: a tensor of shape (batch, input_seq_len,
        units)containing the outputs of the encoder
        :return: y, s
        y is a tensor of shape (batch, vocab) containing the output word as a
        one hot vector in the target vocabulary
        s is a tensor of shape (batch, units) containing the new decoder
        hidden state
        """
        context_vector, attention_weights = self.attention(s_prev,
                                                           hidden_states)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.F(output)
        return x, state
