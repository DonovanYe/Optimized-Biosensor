import torch
import torch.nn as nn

class NonlinearRegressor(nn.Module):
  def __init__(self, input_size):
    def init_weights(m):
      if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    super(NonlinearRegressor, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Linear(input_size, 100),
      nn.BatchNorm1d(100),
      nn.LeakyReLU(0.1),
    )
    self.layer1.apply(init_weights)

    self.layer2 = nn.Sequential(
      nn.Linear(100, 20),
      nn.BatchNorm1d(20),
      nn.LeakyReLU(0.1),
    )
    self.layer2.apply(init_weights)

    self.layer3 = nn.Sequential(
      nn.Linear(20, 1)
    )
       
  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    return out