import pandas as pd
import numpy as np

TRAIN_FILE = "../data/train_regression.parquet"
TEST_FILE = "../data/test_regression.parquet"

# Gives back data in the form
# train, val, test
# where val is None if train_test_split is None
# and train, val, and test are tuples of (X, Y)
def load_np(normalize=True, train_test_split=None, seed=1):
    np.random.seed(seed)

    train_data = pd.read_parquet(TRAIN_FILE).to_numpy()
    test_data = pd.read_parquet(TEST_FILE).to_numpy()

    np.random.shuffle(train_data)
    Xtrain_unnorm = train_data[:, :-1]
    Ytrain_unnorm = train_data[:, -1:]

    Xval_unnorm = None
    Yval_unnorm = None

    if train_test_split:
        train_size = int(train_test_split * len(train_data))
        Xtrain_unnorm = Xtrain_unnorm[:train_size]
        Xval_unnorm = Xtrain_unnorm[train_size:]
        Ytrain_unnorm = Ytrain_unnorm[:train_size]
        Yval_unnorm = Ytrain_unnorm[train_size:]

    Xtest_unnorm = test_data[:, :-1]
    Ytest_unnorm = test_data[:, -1:]

    if not normalize:
        return (Xtrain_unnorm, Ytrain_unnorm), (Xval_unnorm, Yval_unnorm), (Xtest_unnorm, Ytest_unnorm)
    
    Xtrain_norm = Xtrain_unnorm
    Ytrain_norm = Ytrain_unnorm
    Xval_norm = Xval_unnorm
    Yval_norm = Yval_unnorm
    Xtest_norm = Xtest_unnorm
    Ytest_norm = Ytrain_unnorm

    Xmean = np.mean(Xtrain_unnorm, axis=0)
    Xstd = np.std(Xtrain_unnorm, axis=0)
    
    for i in range(len(Xmean)):
        Xtrain_norm[:, i] = (Xtrain_unnorm[:, i] - Xmean[i]) / Xstd[i]
        if Xval_norm:
            Xval_norm[:, i] = (Xval_unnorm[:, i] - Xmean[i]) / Xstd[i]
        Xtest_unnorm[:, i] = (Xtrain_unnorm[:, i] - Xmean[i]) / Xstd[i]
    
    return (Xtrain_norm, Ytrain_norm), (Xval_norm, Yval_norm), (Xtest_norm, Ytest_norm)