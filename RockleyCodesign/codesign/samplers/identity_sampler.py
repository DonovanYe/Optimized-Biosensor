import torch
import torch.nn as nn

class IdentitySampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, measurements):
        return measurements
