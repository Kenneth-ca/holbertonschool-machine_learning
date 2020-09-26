#!/usr/bin/env python3
"""
Creates encoder for transformer
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
    Class to create an encoder for a transformer
    """

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Class constructor
        :param dm: an integer representing the dimensionality of the model
        :param h: an integer representing the number of heads
        :param hidden: the number of hidden units in the fully connected layer
        :param drop_rate: the dropout rate
        """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden,
                                    drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
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

        x = self.embedding(x)
        x = x * tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x = x + self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, encoder_output,
                               training,look_ahead_mask, padding_mask)

        return x
