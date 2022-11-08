import numpy as np
import pandas as pd
import typing


NUM_LASERS = 197


def _process_data(
    data: pd.DataFrame,
    standardize: bool
) -> typing.Tuple[np.array, np.array]:
    """
    Split data into lasers X and ethanol concentration y. Standardize the
    input data to have zero mean and one standard deviation if specified.
    """
    X = data.iloc[:, :NUM_LASERS].to_numpy(copy=True)
    y = data.iloc[:, NUM_LASERS].to_numpy(copy=True)

    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    return X, y

def _standardize_given(
    data: np.array,
    mean: np.array,
    std: np.array,
) -> np.array:
    norm = np.array(data)
    for i in range(len(mean)):
        norm[:, i] = (norm[:, i] - mean[i]) / std[i]
    return norm

def _split_data(
    X: np.array,
    y: np.array,
    split: float,
    seed: int
) -> typing.Tuple[typing.Tuple[np.array, np.array], typing.Tuple[np.array, np.array]]:
    """
    Split data into training and validation sets
    """
    np.random.seed(seed)
    indicies = np.random.permutation(X.shape[0])
    num_train = int(split * X.shape[0])
    train_indicies = indicies[:num_train]
    valid_indicies = indicies[num_train:]

    X_train = X[train_indicies, :]
    y_train = y[train_indicies]

    X_valid = X[valid_indicies, :]
    y_valid = y[valid_indicies]

    return (X_train, y_train), (X_valid, y_valid)


def load_data(
    filename: str,
    standardize: bool = True,
    split: float = 0.8,
    seed: int = 2022
) -> typing.Tuple[typing.Tuple[np.array, np.array], typing.Tuple[np.array, np.array]]:
    """
    Load data from a file and split into training/validation sets.

    Args:
        filename: name of the data file 
        split: percentage of data used for training (remainder used for validation)
        standardize: if true, input features are normalized to have zero mean
        and one standard deviation
        seed: random seed used for data splitting

    Returns:
        (X_train, y_train), (X_valid, y_valid)
    """
    # load data
    data = pd.read_parquet(filename)
    X, y = _process_data(data, standardize)

    return _split_data(X, y, split, seed)

def load_train_test_val(
    trainfile: str,
    testfile: str,
    standardize: bool,
    split: float = 0.8,
    seed: int = 2022,
    precision: int = 64,
    truncate: float = 1,
) -> typing.Tuple[typing.Tuple[np.array, np.array], typing.Tuple[np.array, np.array], typing.Tuple[np.array, np.array]]:
    train, val = load_data(filename=trainfile, standardize=False, split=split, seed=seed)
    test, _ = load_data(filename=testfile, standardize=False, split=1, seed=seed)

    Xtrain_un, Ytrain = train
    Xval_un, Yval = val
    Xtest_un, Ytest = test

    if not standardize:
        if precision == 32:
            Xtrain_un = np.array(Xtrain_un, dtype=np.float32)
            Ytrain = np.array(Ytrain, dtype=np.float32)
            Xval_un = np.array(Xval_un, dtype=np.float32)
            Yval = np.array(Yval, dtype=np.float32)
            Xtest_un = np.array(Xtest_un, dtype=np.float32)
            Ytest = np.array(Ytest, dtype=np.float32)
        
        Xtrain_un = Xtrain_un[:int(truncate * len(Xtrain_un))]
        Ytrain = Ytrain[:int(truncate * len(Ytrain))]
        
        return (Xtrain_un, Ytrain), (Xval_un, Yval), (Xtest_un, Ytest)
    
    Xmean = np.mean(Xtrain_un, axis=0)
    Xstd = np.std(Xtrain_un, axis=0)
    Xtrain_n = _standardize_given(Xtrain_un, Xmean, Xstd)
    Xval_n = _standardize_given(Xval_un, Xmean, Xstd)
    Xtest_n = _standardize_given(Xtest_un, Xmean, Xstd)

    if precision == 32:
        Xtrain_n = np.array(Xtrain_n, dtype=np.float32)
        Ytrain = np.array(Ytrain, dtype=np.float32)
        Xval_n = np.array(Xval_n, dtype=np.float32)
        Yval = np.array(Yval, dtype=np.float32)
        Xtest_n = np.array(Xtest_n, dtype=np.float32)
        Ytest = np.array(Ytest, dtype=np.float32)
    
    Xtrain_n = Xtrain_n[:int(truncate * len(Xtrain_n))]
    Ytrain = Ytrain[:int(truncate * len(Ytrain))]

    return (Xtrain_n, Ytrain), (Xval_n, Yval), (Xtest_n, Ytest)