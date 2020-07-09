#!/usr/bin/env python3

from align import FaceAlign
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import load_images, save_images

fa = FaceAlign('models/landmarks.dat')
images, filenames = load_images('HBTN', as_array=False)
anchors = np.array([[0.194157, 0.16926692], [0.7888591, 0.15817115], [0.4949509, 0.5144414]], dtype=np.float32)
aligned = []
for image in images:
    aligned.append(fa.align(image, np.array([36, 45, 33]), anchors, 96))
aligned = np.array(aligned)
print(aligned.shape)
if not os.path.isdir('HBTNaligned'):
    print(save_images('HBTNaligned', aligned, filenames))
    os.mkdir('HBTNaligned')
print(save_images('HBTNaligned', aligned, filenames))
print(os.listdir('HBTNaligned'))
image = plt.imread('HBTNaligned/KirenSrinivasan.jpg')
plt.imshow(image)
plt.show()
