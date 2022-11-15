import torch,torchvision, pathlib
from torch.utils.data import Subset, Dataset, random_split, TensorDataset
from typing import List, Optional, Union
from torchvision import transforms
from dataclasses import dataclass
from codesign.data.base import BaseDataModule

# Remember to add relevant packages to python path
# import os
# os.environ['PYTHONPATH'] += '~/Desktop/Optimized-Biosensor/RockleyCodesign'
# os.environ['PYTHONPATH'] += '~/Desktop/Optimized-Biosensor/rockley'

import utils.data as data_loader
import pandas as pd
import numpy as np


class RockleyData(Dataset):
    
    def __init__(
        self,
        xdata: np.ndarray,
        ydata: np.ndarray,
    ):
        # self.data = data
        # self.tensor_data = torch.tensor(data).type(torch.FloatTensor)
        self.xtensor = torch.tensor(xdata).type(torch.FloatTensor)
        self.ytensor = torch.tensor(
            ydata.reshape((-1, 1))
        ).type(torch.FloatTensor)

    def __len__(self):
        return len(self.xtensor)

    def __getitem__(self, i: int):
        x = self.xtensor[i]
        y = self.ytensor[i]
        # Seems like this is X, regression val, and label
        return x, y, 0 if y[0] < 80 else 1

@dataclass
class RockleyDataModule(BaseDataModule):
    # Root location relative to where RockleyCodesign folder is
    TRAIN_LOCATION = "../data/train_regression.parquet"
    TEST_LOCATION = "../data/test_regression.parquet"
    VAL_RATIO = 0.2
    
    def __init__(self, shape, batch_size=256, truncate=1, top_idx=-1):
        super().__init__(shape, batch_size)

        train, val, test = data_loader.load_train_test_val(
            trainfile=RockleyDataModule.TRAIN_LOCATION,
            testfile=RockleyDataModule.TEST_LOCATION,
            standardize=True,
            split=(1 - RockleyDataModule.VAL_RATIO),
            truncate=truncate,
            top_idx=top_idx
        ) 

        self.train = train
        self.val = val
        self.test = test

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_set = RockleyData(self.train[0], self.train[1])
            self.val_set = RockleyData(self.val[0], self.val[1])
            # self.train_val_set = RockleyData(self.train_data)
            # val_size = int(len(self.train_val_set) * RockleyDataModule.VAL_RATIO)
            # train_size = len(self.train_val_set) - val_size
            # # Split with fixed seed for reproducibility
            # self.train_set, self.val_set = random_split(self.train_val_set, [train_size, val_size])

        if stage in (None, "test"):
            self.test_set = RockleyData(self.test[0], self.test[1])