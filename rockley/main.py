from rockley.utils.data import load_data


def main():
    (X_train, y_train), (X_valid, y_valid) = load_data("data/train_regression.parquet")
    print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
