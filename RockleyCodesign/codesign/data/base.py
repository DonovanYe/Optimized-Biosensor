from abc import ABC
from pytorch_lightning import LightningDataModule
from dataclasses import dataclass
from torch.utils.data import DataLoader
import numpy as np

@dataclass
class BaseDataModule(LightningDataModule, ABC):
    def __init__(self, shape, batch_size):
        super().__init__()
        self.shape = shape
        self.batch_size = batch_size

    def train_dataloader(self):
        print(f'-------> training set: {int(np.ceil(len(self.train_set)/self.batch_size))} batches of size {self.batch_size} ({len(self.train_set)} samples in total) <-------')
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=2, shuffle=True, drop_last=False)

    def val_dataloader(self):
        print(f'-------> validation set: {len(self.val_set)} batches of size 1 ({len(self.val_set)} samples in total) <-------')
        return DataLoader(self.val_set, batch_size=1, num_workers=2, shuffle=False, drop_last=False)

    def test_dataloader(self):
        print(f'-------> test set: {len(self.test_set)} batches of size 1 ({len(self.test_set)} samples in total) <-------')
        return DataLoader(self.test_set, batch_size=1, num_workers=2, shuffle=False, drop_last=False)

    def teardown(self, stage=None):
        # Used to clean-up when the run is finished
        pass
