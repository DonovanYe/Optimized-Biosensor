import numpy as np

# Normalize Data
def normalize(X_train, X_test, Y_train):
  X_mean = np.mean(X_train, axis=0)
  X_std = np.std(X_train, axis=0)

  X_train_norm = np.array(X_train)
  X_test_norm = np.array(X_test)
  for i in range(len(X_mean)):
    X_train_norm[:, i] = (X_train[:, i] - X_mean[i]) / X_std[i]
    X_test_norm[:, i] = (X_test[:, i] - X_mean[i]) / X_std[i]

  Y_mean = np.mean(Y_train)
  Y_std = np.std(Y_train)
  Y_train_norm = (Y_train - Y_mean) / Y_std

  return X_train_norm, X_test_norm, Y_train_norm

def normalize_one(X_train):
  X_mean = np.mean(X_train, axis=0)
  X_std = np.std(X_train, axis=0)

  X_train_norm = np.array(X_train)
  for i in range(len(X_mean)):
    X_train_norm[:, i] = (X_train[:, i] - X_mean[i]) / X_std[i]

  return X_train_norm

def denormalize(Y_train, Y_test):
  Y_mean = np.mean(Y_train)
  Y_std = np.std(Y_train)
  return Y_test * Y_std + Y_mean