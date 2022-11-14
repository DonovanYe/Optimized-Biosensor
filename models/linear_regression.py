# Training functions

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from util import normalize, denormalize
from plot_util import plot_pred_v_actual

# Train with cross-validation
def train_with_CV(X, Y, model, metric, folds=5, norm=False, graph=True, **kwargs):
  metrics = []
  kf = KFold(n_splits=folds)
  for train, test in kf.split(X):
    x_train, y_train = X[train], Y[train]
    x_test, Y_test = X[test], Y[test]
    if norm:
      X_train, X_test, Y_train = normalize(x_train, x_test, y_train)
    else:
      X_train, X_test, Y_train = x_train, x_test, y_train

    reg = model(**kwargs)
    reg.fit(X_train, Y_train)

    Y_pred = reg.predict(X_test)
    if norm:
      Y_pred = denormalize(y_train, Y_pred)
    m = metric(Y_test, Y_pred)
    metrics.append(m)
  # Take last fold and graph its result on test split and last seen regression
  if graph:
    plot_pred_v_actual(Y_pred, Y_test)
  return np.mean(metrics), reg

# Train just by splitting up the data into train/test
def train_with_split(X, Y, model, metric, split_size=0.2, norm=False, graph=True, **kwargs):
  x_train_split, x_test_split, y_train_split, Y_test_split = train_test_split(X, Y, test_size=0.2)
  if norm:
    X_train_split, X_test_split, Y_train_split = normalize(x_train_split, x_test_split, y_train_split)
  else:
    X_train_split, X_test_split, Y_train_split = x_train_split, x_test_split, y_train_split

  reg = model(**kwargs)
  reg.fit(X_train_split, Y_train_split)

  Y_pred = reg.predict(X_test_split)
  if norm:
    Y_pred = denormalize(y_train_split, Y_pred)
  if graph:
    plot_pred_v_actual(Y_pred, Y_test_split)
  return metric(Y_test_split, Y_pred), reg