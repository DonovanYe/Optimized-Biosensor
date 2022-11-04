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
    return (data - mean) / std

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
    X, y, _ = _process_data(data, standardize)

    return _split_data(X, y, split, seed)

def load_train_test_val(
    trainfile: str,
    testfile: str,
    standardize: bool,
    split: float = 0.8,
    seed: int = 2022,
) -> typing.Tuple[typing.Tuple[np.array, np.array], typing.Tuple[np.array, np.array], typing.Tuple[np.array, np.array]]:
    train, val = load_data(filename=trainfile, standardize=False, split=split, seed=seed)
    test = load_data(filename=testfile, standardize=False, split=1, seed=seed)

    Xtrain_un, Ytrain = train
    Xval_un, Yval = val
    Xtest_un, Ytest = test

    Xmean = np.mean(Xtrain_un, axis=0)
    Xstd = np.std(Xtrain_un, axis=0)
    Xtrain_n = _process_data(Xtrain_un, Xmean, Xstd)
    Xval_n = _process_data(Xval_un, Xmean, Xstd)
    Xtest_n = _process_data(Xtest_un, Xmean, Xstd)

    return (Xtrain_n, Ytrain), (Xval_n, Yval), (Xtest_n, Ytest)