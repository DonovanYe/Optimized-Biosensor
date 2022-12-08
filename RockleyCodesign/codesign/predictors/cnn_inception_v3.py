import torch
import torch.nn as nn


class CNNInceptionV3(nn.Module):
  def __init__(self, input_size):
    def init_weights(m):
      if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    super(CNNInceptionV3, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(input_size, 100, kernel_size=7, stride=3),
      nn.Conv2d(100, 100, kernel_size=7, stride=3),
      nn.Conv2d(100, 100, kernel_size=7, stride=3),
      nn.Conv2d(100, 100, kernel_size=7, stride=3),
      nn.Conv2d(100, 100, kernel_size=7, stride=3),
      nn.Conv2d(100, 100, kernel_size=7, stride=3),
      nn.Conv2d(100, 100, kernel_size=7, stride=3),
      nn.Conv2d(100, 100, kernel_size=7, stride=3),
    )
    self.layer1.apply(init_weights)

    self.layer2 = nn.Sequential(
      nn.Conv2d(input_size, 20, kernel_size=7, stride=3),
      nn.Conv2d(20, 20, kernel_size=7, stride=3),
      nn.Conv2d(20, 20, kernel_size=7, stride=3),
      nn.Conv2d(20, 20, kernel_size=7, stride=3),

      nn.Conv2d(20, 20, kernel_size=5, stride=1),
      nn.Conv2d(20, 20, kernel_size=5, stride=1),
      nn.Conv2d(20, 20, kernel_size=5, stride=1),
      nn.Conv2d(20, 20, kernel_size=5, stride=1),

      nn.Conv2d(20, 20, kernel_size=3, stride=1),
      nn.Conv2d(20, 20, kernel_size=3, stride=1),
      nn.Conv2d(20, 20, kernel_size=3, stride=1),
      nn.Conv2d(20, 20, kernel_size=3, stride=1),
    )
    self.layer2.apply(init_weights)

    self.layer3 = nn.Sequential(
      nn.Linear(20, 1)
    )

    self.f1 = nn.Sequential(
      nn.Linear(?, 20)
    )

    self.f2 = nn.Sequential(
      nn.Linear(?, 20),
      nn.Dropout2d(0.2)
    )
       
  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = torch.flatten(out)
    out = self.f1(out)
    return out