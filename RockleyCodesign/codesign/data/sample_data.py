import torch,torchvision, pathlib
from torch.utils.data import Subset, Dataset, random_split, TensorDataset
from typing import List, Optional, Union
from torchvision import transforms
from dataclasses import dataclass
from codesign.data.base import BaseDataModule

import pandas as pd
import numpy as np


class SampleData(Dataset):
    
    def __init__(
        self,
        data: np.ndarray
    ):
        self.data = data
        self.tensor_data = torch.tensor(data).type(torch.FloatTensor)
        self.x = data[:, :-1]
        self.y = data[:, -1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int):
        x = self.tensor_data[i, :-1]
        y = self.tensor_data[i, -1:]
        # Seems like this is X, regression val, and label
        # But the label cannot be [] for some reason???
        # Shape?
        return x, y, 0 if y[0] < 90 else 1

@dataclass
class SampleDataModule(BaseDataModule):
    VAL_RATIO = 0.2
    
    def __init__(self, shape, batch_size=128):
        super().__init__(shape, batch_size)
        size = 10000
        features = 10
        train_data = np.random.rand(size, features + 1)
        test_data = np.random.rand(size, features + 1)
        for i in range(size):
            train_data[i, -1] = 0
            test_data[i, -1] = 0
            for j in range(features):
                train_data[i, -1] += (j + 1) * train_data[i, j]
                test_data[i, -1] += (j + 1) * test_data[i, j]
        self.train_data = train_data
        self.test_data = test_data

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_val_set = SampleData(self.train_data)
            val_size = int(len(self.train_val_set) * SampleDataModule.VAL_RATIO)
            train_size = len(self.train_val_set) - val_size
            # Split with fixed seed for reproducibility
            self.train_set, self.val_set = random_split(self.train_val_set, [train_size, val_size])

        if stage in (None, "test"):
            self.test_set = SampleData(self.test_data)