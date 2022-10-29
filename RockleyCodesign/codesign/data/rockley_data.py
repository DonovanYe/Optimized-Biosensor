import torch,torchvision, pathlib
from torch.utils.data import Subset, Dataset, random_split, TensorDataset
from typing import List, Optional, Union
from torchvision import transforms
from dataclasses import dataclass
from codesign.data.base import BaseDataModule

import pandas as pd
import numpy as np


class RockleyData(Dataset):
    
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
class RockleyDataModule(BaseDataModule):
    # Root location relative to where RockleyCodesign folder is
    TRAIN_LOCATION = "../data/train_data_normalized.npy"
    TEST_LOCATION = "../data/test_data_normalized.npy"
    VAL_RATIO = 0.2
    
    def __init__(self, shape, batch_size=128):
        super().__init__(shape, batch_size)
        self.train_data = np.load(RockleyDataModule.TRAIN_LOCATION)
        self.test_data = np.load(RockleyDataModule.TEST_LOCATION)

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_val_set = RockleyData(self.train_data)
            val_size = int(len(self.train_val_set) * RockleyDataModule.VAL_RATIO)
            train_size = len(self.train_val_set) - val_size
            # Split with fixed seed for reproducibility
            self.train_set, self.val_set = random_split(self.train_val_set, [train_size, val_size])

        if stage in (None, "test"):
            self.test_set = RockleyData(self.test_data)