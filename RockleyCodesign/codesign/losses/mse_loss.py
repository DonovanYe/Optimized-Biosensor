import torch.nn as nn
from torch.nn import MSELoss as TorchMSELoss

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = TorchMSELoss()

    @property
    def name(self):
        return "mse_loss"

    @property
    def mode(self):
        return "min"

    @property
    def task(self):
        return "reg"

    def forward(self, pred, gt):
        return self.loss(pred, gt)

class MSE(MSELoss):
    @property
    def name(self):
        return "mse"
