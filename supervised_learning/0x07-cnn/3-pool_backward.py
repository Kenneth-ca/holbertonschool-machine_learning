#!/usr/bin/env python3
"""
Performs backward propagation over a convolutional layer of a neural network
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    a function that performs backward propagation over a CNN
    :param dA: is a numpy.ndarray of shape (m, h_new, w_new, c_new)
    containing the partial derivatives with respect to the output of the
    pooling layer
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c is the number of channels
    :param A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
    :param kernel_shape: is a tuple of (kh, kw) containing the size of the
    kernel for the pooling
        kh is the kernel height
        kw is the kernel width
    :param stride: is a tuple of (sh, sw) containing the strides for the
    convolution
    :param mode: is a string containing either max or avg, indicating whether
    to perform maximum or average pooling, respectively
    :return: the partial derivatives with respect to the previous layer (
    dA_prev)
    """
    m, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    # In this notation row refers to height and col to width
    for img in range(m):
        A_img = A_prev[img]
        for row in range(h_new):
            for col in range(w_new):
                for ch in range(c_new):
                    # corners of the slice
                    row_start = row * sh
                    row_end = row * sh + kh
                    col_start = col * sw
                    col_end = col * sw + kw

                    slice_A = A_img[row_start:row_end, col_start:col_end, ch]
                    if mode == "max":
                        mask = (slice_A == np.max(slice_A)).astype(int)
                        aux = dA[img, row, col, ch] * mask
                        dA_prev[img, row_start:row_end, col_start:col_end,
                                ch] += aux
                    if mode == "avg":
                        average = dA[img, row, col, ch] / (kh * kw)
                        mask = np.ones(kernel_shape) * average
                        dA_prev[img, row_start:row_end, col_start:col_end,
                                ch] += mask
    return dA_prev
