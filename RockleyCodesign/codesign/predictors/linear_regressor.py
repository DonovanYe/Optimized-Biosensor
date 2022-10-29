import torch.nn as nn
import torch
import numpy as np

class LinearRegressor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layer = nn.Linear(input_size, 1)
        # actual = np.load("../data/actual_weights.npy")
        # self.layer.weight.data[...] = torch.Tensor(actual + np.random.rand(197) * 200)

    def forward(self, measurements_selected):
        out = self.layer(measurements_selected)
        return out
