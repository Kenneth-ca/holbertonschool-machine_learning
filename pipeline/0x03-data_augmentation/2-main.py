#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
rotate_image = __import__('2-rotate').rotate_image

tf.compat.v1.enable_eager_execution()
tf.compat.v1.set_random_seed(2)

doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
for image, _ in doggies.shuffle(10).take(1):
    plt.imshow(rotate_image(image))
    plt.show()
