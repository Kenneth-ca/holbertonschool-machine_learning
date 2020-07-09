#!/usr/bin/env python3

from align import FaceAlign
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

fa = FaceAlign('models/landmarks.dat')
test_img = mpimg.imread('HBTN/KirenSrinivasan.jpg')
box = fa.detect(test_img)
landmarks = fa.find_landmarks(test_img, box)
print(type(landmarks), landmarks.shape)
plt.imshow(test_img)
ax = plt.gca()
for landmark in landmarks:
    ax.add_patch(Circle(landmark))
plt.show()
