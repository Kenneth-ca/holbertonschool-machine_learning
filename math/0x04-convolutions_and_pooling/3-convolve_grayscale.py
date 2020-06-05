#!/usr/bin/env python3
"""
Performs a strided convolution on grayscale images
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    a function that performs a strided convolution
    :param images: a numpy.ndarray with shape (m, h, w) containing multiple
    grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    :param kernel: a numpy.ndarray with shape (kh, kw) containing the kernel
    for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    :param padding: a tuple of (ph, pw)
        ph is the padding for the height of the image
        pw is the padding for the width of the image
    :param stride: is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    :return: numpy.ndarray containing the convolved images
    """
    c_images = images.shape[0]
    f_height = kernel.shape[0]
    f_width = kernel.shape[1]

    if padding == "same":
        if f_height % 2 != 0:
            padding_h = (f_height - 1) // 2
        else:
            padding_h = f_height // 2
        if f_width % 2 != 0:
            padding_w = (f_width - 1) // 2
        else:
            padding_w = f_width // 2
    elif padding == "valid":
        padding_h, padding_w = (0, 0)
    else:
        padding_h, padding_w = padding
    stride_h, stride_w = stride

    c_height = (images.shape[1] + 2 * padding_h - f_height) // stride_h + 1
    c_width = (images.shape[2] + 2 * padding_w - f_width) // stride_w + 1
    # np.pad works with a before_N and after_N parameter defined in a tuple
    # that will add the selected pad at each dimension
    pad_images = np.pad(images, ((0, 0), (padding_h, padding_h), (padding_w,
                                                                  padding_w)))

    convolved = np.zeros((c_images, c_height, c_width))
    for row in range(c_height):
        for col in range(c_width):
            pad_ele = pad_images[:, row * stride_h:row * stride_h + f_height,
                                 col * stride_w:col * stride_w + f_width]
            sum_mul_ele = np.sum(pad_ele * kernel, axis=(1, 2))
            convolved[:, row, col] = sum_mul_ele
    return convolved
