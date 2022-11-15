import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Indices in order of removal
def removed_matrix(idxs, colors=['tomato', 'springgreen']):
  X = np.ones((197, 197))
  X[0, idxs[0]] = 0
  for i in range(1, len(idxs)):
    X[:, i] = X[:, i - 1]
    X[idxs[i], i] = 0

  cmap = LinearSegmentedColormap.from_list('my_cmap', ['tomato', 'springgreen'])
  plt.imshow(X=X, cmap=cmap)
  plt.xlabel("Present at step N")
  plt.ylabel("Laser No.")
  plt.title("Removal Matrix")
  plt.show()