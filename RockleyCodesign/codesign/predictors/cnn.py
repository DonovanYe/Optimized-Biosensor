import torch
import torch.nn as nn

class CNNRegressor(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=1, stride=2),
            nn.LeakyReLU(0.2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=5, stride=2),
            nn.LeakyReLU(0.2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(4 * 72, 16),
        )
        self.fc1.apply(init_weights)
        self.fc2 = nn.Linear(in_features=16, out_features=1)
        self.fc2.apply(init_weights)

    def forward(self, measurements_selected):
        measurements_selected = torch.unsqueeze(measurements_selected, -2)
        x1 = self.layer1(measurements_selected)
        x2 = self.layer2(measurements_selected)
        x3 = self.layer3(measurements_selected)
        xs = torch.cat((x1, x2, x3), dim=-1)
        flat = torch.flatten(xs, start_dim=1)
        out = self.fc1(flat)
        out = self.fc2(out)
        return out
        # out = self.layer1(measurements_selected)
        # out = torch.unsqueeze(out, -2)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = out.view(out.size(0), -1)
        # out = self.fc1(out)
        # out = self.fc2(out)

        return out