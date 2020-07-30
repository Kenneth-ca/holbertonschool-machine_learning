#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
tsne = __import__('8-tsne').tsne

np.random.seed(0)
X = np.loadtxt("mnist2500_X.txt")
labels = np.loadtxt("mnist2500_labels.txt")
Y = tsne(X, perplexity=50.0, iterations=3000, lr=750)
plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
plt.colorbar()
plt.title('t-SNE')
plt.show()
