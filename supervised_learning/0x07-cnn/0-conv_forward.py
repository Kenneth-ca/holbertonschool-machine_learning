#!/usr/bin/env python3
"""
Performs forward propagation over a convolutional layer of a neural network
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    a function that performs forward propagation over a CNN
    :param A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
    :param W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
    kernels for the convolution
    :param b: numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
    applied to the convolution
    :param activation: is an activation function applied to the convolution
    :param padding:  is a string that is either same or valid, indicating the
    type of padding used
    :param stride: is a tuple of (sh, sw) containing the strides for the
    convolution
    :return: the output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    pad_h, pad_w = (0, 0)
    if padding == "same":
        pad_h = int(np.ceil((((h_prev - 1) * sh + kh - h_prev) / 2)))
        pad_w = int(np.ceil((((w_prev - 1) * sw + kw - w_prev) / 2)))

    padded = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                    mode="constant", constant_values=(0, 0))

    conv_h = (h_prev + 2 * pad_h - kh) // sh + 1
    conv_w = (w_prev + 2 * pad_w - kw) // sw + 1
    convolved = np.zeros((m, conv_h, conv_w, c_new))

    # In this notation row refers to height and col to width
    for row in range(conv_h):
        for col in range(conv_w):
            for ch in range(c_new):
                slice_A = padded[:, row * sh:row * sh + kh, col * sw:col * sw
                                 + kw]
                slice_A_sum = np.sum(slice_A * W[:, :, :, ch], axis=(1, 2, 3))
                convolved[:, row, col, ch] = slice_A_sum
    return activation(convolved + b)
