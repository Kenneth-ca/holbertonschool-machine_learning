#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

# Figure 3D
fig = plt.figure()
axes = Axes3D(fig)

shift = np.stack(pca_data, axis=-1)
x, y, z = shift
axes.scatter(x, y, z, c=labels, cmap=plt.cm.plasma)

axes.set_xlabel("U1")
axes.set_ylabel("U2")
axes.set_zlabel("U3")
plt.title("PCA of Iris Dataset")
plt.show()
