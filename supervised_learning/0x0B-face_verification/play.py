#!/usr/bin/env python3
import re
import numpy as np
from verification import FaceVerification
from utils import load_images
import tensorflow as tf
import matplotlib.pyplot as plt
import random
# image loading (15)
images, filenames = load_images('HBTNaligned', as_array=True)
identities = [re.sub('[0-9]', '', f[:-4]) for f in filenames]
# model loading
with tf.keras.utils.CustomObjectScope({'tf': tf}):
    my_model = tf.keras.models.load_model('models/trained_fv.h5')
embedded = np.zeros((images.shape[0], 128))
for i, img in enumerate(images):
    embedded[i] = my_model.predict(np.expand_dims(img, axis=0))[0]
database = np.array(embedded)
fv = FaceVerification('models/trained_fv.h5', database, identities)
# image selection
test_images, filenames = load_images('TESTaligned', as_array=True)
my_image = test_images[0]
identity, distance = fv.verify(my_image)
print(identity, distance)
if identity is None:
    identity = 'Not recognized'
plt.imshow(my_image)
plt.title('Recognized as ' + identity)
plt.show()
