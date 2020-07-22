#!/usr/bin/env python3

from align import FaceAlign
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fa = FaceAlign('models/landmarks.dat')
test_img = mpimg.imread('HBTN/KirenSrinivasan.jpg')
box = fa.detect(test_img)
print(type(box))
plt.imshow(test_img)
ax = plt.gca()
rect = Rectangle((box.left(), box.top()), box.width(), box.height(), fill=False)
ax.add_patch(rect)
plt.show()
