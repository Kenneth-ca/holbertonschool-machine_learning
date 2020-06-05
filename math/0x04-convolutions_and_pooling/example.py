#!/usr/bin/env python3
"""
File to make tests with numpy.sum
"""
import numpy as np


arr = np.array([[[5, 1, 0], [3, 2, 1]],
              [[1, 2, 3], [1, 1, 0]]])
print(arr)
# The sum specifying all axis is the np.sum by default
sum0 = np.sum(arr, axis=(0, 1, 2))
print("sum all elements: ", sum0)
# You can sum along axis
sum1 = np.sum(arr, axis=0)
print("sum along axis 0: ", sum1)
# And along two axis at a time
sum2 = np.sum(arr, axis=(0, 1))
print("sum axis 0 and 1: ", sum2)

# Padding experiments...

a = np.array([[[5, 1, 0], [3, 2, 1]],
              [[1, 2, 3], [1, 1, 0]]])
print(a)
print(a.shape)
print("*******")
print(np.pad(a, ((1, 1), (0, 0), (0, 0))))
print("*******")
print(np.pad(a, ((0, 0), (1, 1), (0, 0))))
print("*******")
print(np.pad(a, ((0, 0), (0, 0), (1, 1))))
