import torch
import torch.nn as nn

class NonlinearRegressor(nn.Module):
  def __init__(self, input_size):
    super(NonlinearRegressor, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Linear(input_size, 20),
      nn.BatchNorm1d(20),
      nn.ReLU(),
    )

    self.layer2 = nn.Sequential(
      nn.Linear(20, 10),
      nn.BatchNorm1d(10),
      nn.ReLU(),
    )

    self.layer3 = nn.Sequential(
      nn.Linear(10, 1)
    )
       
  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    return out