#!/usr/bin/env python3
"""
Creates all masks for training/validation
"""
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """
    Creates all masks for training/validation
    :param inputs: a tf.Tensor of shape (batch_size, seq_len_in) that
    contains the input sentence
    :param target: is a tf.Tensor of shape (batch_size, seq_len_out) that
    contains the target sentence
    :return: encoder_mask, look_ahead_mask, decoder_mask
        encoder_mask is the tf.Tensor padding mask of shape (batch_size, 1,
        1, seq_len_in) to be applied in the encoder
        look_ahead_mask is the tf.Tensor look ahead mask of shape (
        batch_size, 1, seq_len_out, seq_len_out) to be applied in the decoder
        decoder_mask is the tf.Tensor padding mask of shape (batch_size, 1,
        1, seq_len_in) to be applied in the decoder
    """
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    size = target.shape[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

    dec_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    dec_mask = dec_mask[:, tf.newaxis, tf.newaxis, :]

    combined_mask = tf.maximum(dec_mask, look_ahead_mask)

    return encoder_mask, combined_mask, decoder_mask
