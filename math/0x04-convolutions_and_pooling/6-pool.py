#!/usr/bin/env python3
"""
Performs pooling on images
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    a function that performs pooling on images
    :param images: a numpy.ndarray with shape (m, h, w) containing multiple
    grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image
    :param kernel_shape: a numpy.ndarray with shape (kh, kw) containing the
    kernel
    for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    :param stride: is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    :param mode: indicates the type of pooling
        max indicates max pooling
        avg indicates average pooling
    :return: numpy.ndarray containing the convolved images
    """
    c_images, images_h, images_w, channels = images.shape
    f_height = kernel_shape[0]
    f_width = kernel_shape[1]
    stride_h, stride_w = stride

    p_height = (images_h - f_height) // stride_h + 1
    p_width = (images_w - f_width) // stride_w + 1

    pooled = np.zeros((c_images, p_height, p_width, channels))
    for row in range(p_height):
        for col in range(p_width):
            ele = images[:, row * stride_h:row * stride_h + f_height,
                         col * stride_w:col * stride_w + f_width]

            if mode == "max":
                pooled[:, row, col] = np.max(ele, axis=(1, 2))
            if mode == "avg":
                pooled[:, row, col] = np.mean(ele, axis=(1, 2))

    return pooled
