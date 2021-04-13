
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNTanh (nn.Module):
    def __init__(self):
        super(CNNTanh, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # x = self.pool(F.tanh(self.conv1(x)))
        # x = self.pool(F.tanh(self.conv2(x)))
        x = F.tanh(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.tanh(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # x = x.view(-1, 16 * 5 * 5)
        x = torch.flatten(x, start_dim=1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
