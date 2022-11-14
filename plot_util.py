import matplotlib.pyplot as plt
import numpy as np

# Plot predicted vs actual graph
def plot_pred_v_actual(Y_pred, Y_actual):
  plt.scatter(Y_actual, Y_pred, s=0.1, color='blue')

  xs = np.linspace(1, 200)
  plt.plot(xs, xs, color='red')

  plt.xlabel("Actual")
  plt.ylabel("Predicted")
  plt.title("Predicted vs. Actual Graph")
  plt.show()