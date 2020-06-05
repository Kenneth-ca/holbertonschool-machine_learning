"""
Performs a same convolution on grayscale images
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    a function that performs a same convolution
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
    padding_h = (kernel.shape[0] - 1) // 2
    padding_w = (kernel.shape[1] - 1) // 2
    c_images = images.shape[0]
    c_height = images.shape[1] + 2 * padding_h - kernel.shape[0] + 1
    c_width = images.shape[2] + 2 * padding_w - kernel.shape[1] + 1
    # np.pad works with a before_N and after_N parameter defined in a tuple
    # that will add the selected pad at each dimension
    pad_images = np.pad(images, ((0, 0), (padding_h, padding_h), (padding_w,
                                                                  padding_w)))
    convolved = np.zeros((c_images, c_height, c_width))
    for row in range(c_height):
        for col in range(c_width):
            element = np.sum(pad_images[:, row:row+3, col:col+3] * kernel,
                             axis=(1, 2))
            convolved[:, row, col] = element
    return convolved
