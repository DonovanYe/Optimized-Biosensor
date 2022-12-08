import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

NUM_LASERS = 197


REVERSE_BEAM_RESULTS = {
    0.0:  [50, 91, 97, 9, 33, 29, 49, 45, 46, 48,
           47, 32, 6, 44, 90, 34, 95, 87, 11, 15,
           191, 26, 141, 30, 192, 103, 8, 111, 31, 99,
           190, 187, 172, 148, 189, 28, 142, 23, 158, 117,
           22, 196, 139, 89, 3, 27, 195, 51, 55, 144],
    1e-5: [50, 91, 97, 9, 33, 29, 47, 53, 96, 17,
           0, 62, 1, 4, 55, 184, 182, 147, 142, 60,
           166, 172, 177, 173, 74, 56, 71, 19, 58, 61,
           2, 92, 30, 100, 34, 70, 68, 66, 22, 28,
           77, 5, 176, 124, 132, 136, 123, 48, 72, 164]
}


def process_data(
    train: np.array,
    test:  np.array,
    split: float = 0.8,
    seed: int = 2022,
    noise_std: float = 0.0
) -> tuple[tuple[np.array, np.array],
           tuple[np.array, np.array],
           tuple[np.array, np.array]]:
    """
    Split data into training and validation and standardize X using the
    training set. Optionally add Gaussian noise.
    """
    X = train.iloc[:, :NUM_LASERS].to_numpy(copy=True, dtype=np.float32)
    y = train.iloc[:, NUM_LASERS].to_numpy(copy=True, dtype=np.float32)

    X_test = test.iloc[:, :NUM_LASERS].to_numpy(copy=True, dtype=np.float32)
    y_test = test.iloc[:, NUM_LASERS].to_numpy(copy=True, dtype=np.float32)

    # split into training and validation
    np.random.seed(seed)
    indicies = np.random.permutation(X.shape[0])
    num_train = int(split * X.shape[0])
    train_indicies = indicies[:num_train]
    valid_indicies = indicies[num_train:]

    X_train = X[train_indicies, :]
    y_train = y[train_indicies]

    X_valid = X[valid_indicies, :]
    y_valid = y[valid_indicies]

    # add noise if specified
    assert noise_std >= 0.0
    if noise_std > 0.0:
        X_train += np.random.normal(scale=noise_std, size=X_train.shape)
        X_valid += np.random.normal(scale=noise_std, size=X_valid.shape)
        X_test  += np.random.normal(scale=noise_std, size=X_test.shape)

    # standardize data
    X_mean = np.mean(X_train, axis=0)
    X_std  = np.std(X_train, axis=0)

    # note: we normalize the validation set using mean/std from
    # the training set
    X_train = (X_train - X_mean) / X_std
    X_valid = (X_valid - X_mean) / X_std
    X_test  = (X_test  - X_mean) / X_std

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def reverse_beam_search(X_train, y_train, iters=50, log=True):
    # contains the indicies of lasers selected by reverse beam
    # in order of selection
    selected_lasers  = []

    # keep track of unselected lasers to search in future iterations
    remaining_lasers = list(range(NUM_LASERS))

    for i in range(iters):

        selected_reg = None
        selected_laser = -1
        min_mse = np.inf
        for laser in remaining_lasers:
            mask = selected_lasers + [laser]
            reg = LinearRegression().fit(X_train[:, mask], y_train)
            y_pred = reg.predict(X_train[:, mask])
            mse = mean_squared_error(y_train, y_pred)
            if mse < min_mse:
                min_mse = mse
                selected_laser = laser
                selected_reg = reg

        selected_lasers.append(selected_laser)
        remaining_lasers.remove(selected_laser)

        if log:
            print(f'{len(selected_lasers)} lasers selected')
            print(f'coef = {selected_reg.coef_}')
            print(f'intercept = {selected_reg.intercept_}')
            print(f'R^2 = {reg.score(X_train[:, selected_lasers], y_train)}')
            print(f'mse = {min_mse}')
            print(f'selected_lasers = {selected_lasers}')
            print('-' * 75)

    return selected_lasers


def ridge_regression(
    X_train, y_train,
    X_valid, y_valid,
    degree=1,
    alpha=0.0,
    log=True,
):
    num_lasers = X_train.shape[1]

    # generate polynomial and interaction features if specified
    assert degree >= 1
    if degree > 1:
        poly = PolynomialFeatures(degree, include_bias=False)
        X_train = poly.fit_transform(X_train)
        X_valid = poly.fit_transform(X_valid)
    
    reg = Ridge(alpha=alpha).fit(X_train, y_train)
    train_mse = mean_squared_error(reg.predict(X_train), y_train)
    valid_mse = mean_squared_error(reg.predict(X_valid), y_valid)

    if log:
        print(f'num lasers = {num_lasers}')
        print(f'alpha = {alpha}')
        print(f'degree = {degree}')
        print(f'train mse = {train_mse}')
        print(f'valid mse = {valid_mse}')
        print('-' * 75)

    return valid_mse


if __name__ == '__main__':
    train = pd.read_parquet('./data/train_regression.parquet')
    test  = pd.read_parquet('./data/test_regression.parquet')

    for std in [0.0, 1e-5]:
        ((X_train, y_train),
         (X_valid, y_valid),
         (X_test, y_test)) = process_data(train, test, noise_std=std)

        # run reverse beam if result is not saved
        if std in REVERSE_BEAM_RESULTS:
            selected_lasers = REVERSE_BEAM_RESULTS[std]
        else:
            selected_lasers = reverse_beam_search(X_train, y_train)

        # perform linear and quadratic regression
        # on reverse beam results
        num_lasers = list(range(11, 21))
        valid_mse = [
            [], []
        ]
        for n in num_lasers:
            print(selected_lasers[:n])
            for i in range(len(valid_mse)):
                mse = ridge_regression(
                    X_train[:, selected_lasers[:n]], y_train,
                    X_valid[:, selected_lasers[:n]], y_valid,
                    degree=i+1,
                )
                valid_mse[i].append(mse)

        for i in range(len(valid_mse)):
            plt.plot(num_lasers, valid_mse[i], label=f'degree {i+1}')
        plt.xlabel('Num Lasers')
        plt.ylabel('Validation MSE')
        plt.title(f'Ridge Regression (noise std = {std})')
        plt.legend()
        plt.show()
