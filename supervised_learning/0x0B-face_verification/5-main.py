#!/usr/bin/env python3

from align import FaceAlign
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

fa = FaceAlign('models/landmarks.dat')
test_img = mpimg.imread('HBTN/KirenSrinivasan.jpg')
anchors = np.array([[0.194157, 0.16926692], [0.7888591, 0.15817115], [0.4949509, 0.5144414]], dtype=np.float32)
aligned = fa.align(test_img, np.array([36, 45, 33]), anchors, 96)
plt.imshow(aligned)
ax = plt.gca()
for anchor in anchors:
    ax.add_patch(Circle(anchor * 96, 1))
plt.show()
