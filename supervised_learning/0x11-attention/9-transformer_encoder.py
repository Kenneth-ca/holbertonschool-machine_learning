#!/usr/bin/env python3
"""
Creates encoder for transformer
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
    Class to create an encoder for a transformer
    """

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Class constructor
        :param dm: an integer representing the dimensionality of the model
        :param h: an integer representing the number of heads
        :param hidden: the number of hidden units in the fully connected layer
        :param drop_rate: the dropout rate
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm

        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, self.dm)

        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]

        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Function to create the decoder block for the transformer
        :param x: a tensor of shape (batch, target_seq_len, dm)containing the
        input to the decoder block
        :param encoder_output: a tensor of shape (batch, input_seq_len, dm)
        containing the output of the encoder
        :param training: a boolean to determine if the model is training
        :param look_ahead_mask: the mask to be applied to the first multi
        head attention layer
        :param padding_mask: the mask to be applied to the second multi head
        attention layer
        :return: a tensor of shape (batch, target_seq_len, dm) containing
        the block’s output
        """
        seq_len = x.shape[1]
        embedding = self.embedding(x)
        embedding = embedding * tf.math.sqrt(tf.cast(self.dm, tf.float32))
        embedding = embedding + self.positional_encoding[:seq_len]

        encoder_out = self.dropout(embedding, training=training)

        for i in range(self.N):
            encoder_out = self.blocks[i](encoder_out, training, mask)

        return encoder_out
