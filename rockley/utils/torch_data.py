import torch
import utils.data as data
import typing
import numpy as np

class RockleyDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, X, y):
        'Initialization'
        self.X = X
        self.y = y

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.X[index], self.y[index]

def generate_dataloaders(
    trainfile: str,
    testfile: str,
    standardize: bool = True,
    split: float = 0.8,
    seed: int = 2022,
    precision: int = 64,
    truncate: float = 1,
    standardize_y: bool = False,
    batch_size: int = 32,
    shuffle: bool = True,
):
    train, val, test = data.load_train_test_val(
        trainfile=trainfile,
        testfile=testfile,
        standardize=standardize,
        split=split,
        seed=seed,
        precision=precision,
        truncate=truncate,
        standardize_y=standardize_y,
    )
    train_dataset = RockleyDataset(train[0], train[1])
    val_dataset = RockleyDataset(val[0], val[1])
    test_dataset = RockleyDataset(test[0], test[1])

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=shuffle, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def generate_one_batch_dataloaders(
    trainfile: str,
    testfile: str,
    standardize: bool = True,
    split: float = 0.8,
    seed: int = 2022,
    precision: int = 64,
    truncate: float = 1,
    standardize_y: bool = False,
    shuffle: bool = True,
):
    train, val, test = data.load_train_test_val(
        trainfile=trainfile,
        testfile=testfile,
        standardize=standardize,
        split=split,
        seed=seed,
        precision=precision,
        truncate=truncate,
        standardize_y=standardize_y,
    )
    train_dataset = RockleyDataset(train[0], train[1])
    val_dataset = RockleyDataset(val[0], val[1])
    test_dataset = RockleyDataset(test[0], test[1])

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=shuffle, batch_size=len(train_dataset))
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=shuffle, batch_size=len(val_dataset))
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=shuffle, batch_size=len(test_dataset))

    return train_loader, val_loader, test_loader

def generate_fake_data():
    X = np.random.rand(10000, 100) * 100
    y = []
    for i in range(len(X)):
        y.append(0)
        for j in range(len(X[i])):
            y[-1] += (2 * j + 1) * X[i, j]
    y = np.array(y, dtype=np.float32)
    X = np.array(X, dtype=np.float32)

    train_dataset = RockleyDataset(X, y)

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=32)
    val_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=len(train_dataset))

    return train_loader, val_loader