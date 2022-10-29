import torchvision, pathlib
from torch.utils.data import Subset, Dataset, random_split
from typing import List, Optional
from torchvision import transforms
from dataclasses import dataclass
from codesign.data.base import BaseDataModule


class FashionMNISTData(Dataset):
    class_names = {
        0: 'T-shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'
    }

    # This is a wrapper of the original FashionMNIST dataset
    # We provide both target images and k-space measurements
    def __init__(
        self,
        data_dir: pathlib.Path,
        shape: List[int],
        custom_split: Optional[str] = None
    ):
        super().__init__()
        transform = transforms.Compose([
            transforms.Resize(shape),
            transforms.ToTensor()
        ])
        self.examples = torchvision.datasets.FashionMNIST(
            root=data_dir,
            train=(custom_split=='train'),
            download=True,
            transform=transform
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        image, label = self.examples[i]

        return image, [], label


@dataclass
class FashionMNISTDataModule(BaseDataModule):
    def __init__(self, shape, batch_size=24):
        super().__init__(shape, batch_size)

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_val_set = FashionMNISTData(
                './datasets',
                self.shape,
                custom_split='train'
            )
            val_size = len(self.train_val_set) // 4
            train_size = len(self.train_val_set) - val_size
            self.train_set, self.val_set = random_split(self.train_val_set, [train_size, val_size])

        if stage in (None, "test"):
            self.test_set = FashionMNISTData(
                './datasets',
                self.shape,
                custom_split='test'
            )
