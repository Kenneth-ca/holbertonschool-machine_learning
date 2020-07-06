#!/usr/bin/env python3

from utils import load_images, load_csv, generate_triplets
import numpy as np
import matplotlib.pyplot as plt

images, filenames = load_images('HBTNaligned', as_array=True)
triplet_names = load_csv('FVTriplets.csv')
A, P, N = generate_triplets(images, filenames, triplet_names)
plt.subplot(1, 3, 1)
plt.title('Anchor:' + triplet_names[0][0])
plt.imshow(A[0])
plt.subplot(1, 3, 2)
plt.title('Positive:' + triplet_names[0][1])
plt.imshow(P[0])
plt.subplot(1, 3, 3)
plt.title('Negative:' + triplet_names[0][2])
plt.imshow(N[0])
plt.tight_layout()
plt.show()
