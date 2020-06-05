#!/usr/bin/env python3
"""
Performs a valid convolution on grayscale images
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    a function that performs a convolution
    :param images: a numpy.ndarray with shape (m, h, w) containing multiple
    grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    :param kernel: a numpy.ndarray with shape (kh, kw) containing the kernel
    for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    :return: numpy.ndarray containing the convolved images
    """
    c_images = images.shape[0]
    f_height = kernel.shape[0]
    f_width = kernel.shape[1]
    c_height = images.shape[1] - f_height + 1
    c_width = images.shape[2] - f_width + 1

    convolved = np.zeros((c_images, c_height, c_width))
    for row in range(c_height):
        for col in range(c_width):
            mul_ele = images[:, row:row + f_height, col:col + f_width] * kernel
            sum_ele = np.sum(mul_ele, axis=(1, 2))
            convolved[:, row, col] = sum_ele
    return convolved
