import numpy as np
from sklearn.decomposition import PCA

X = np.array([[24, 13, 38], [8, 3, 17], [21, 6, 40], [1, 14, -9], [9, 3, 21], [7, 1, 14],
              [8, 7, 11], [10, 16, 3], [1, 3, 2], [15, 2, 30], [4, 6, 1], [12, 10, 18], [1, 9, -4],
              [7, 3, 19], [5, 1, 13], [1, 12, -6], [21, 9, 34], [8, 8, 7], [1, 18, -18],
              [15, 8, 25], [16, 10, 29], [7, 0, 17], [14, 2, 31], [3, 7, 0], [5, 6, 7]])

pca = PCA(n_components=1)
pca.fit(X)

## New code below
p = pca.components_
centroid = np.mean(X, 0)
segments = np.arange(-40, 40)[:, np.newaxis] * p

import matplotlib
matplotlib.use('TkAgg') # might not be necessary for you
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatterplot = ax.scatter(*(X.T))
lineplot = ax.plot(*(centroid + segments).T, color="red")
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('resultPCA.png', dpi=150)