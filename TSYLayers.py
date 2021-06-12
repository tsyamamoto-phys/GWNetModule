"""
TSYLayers.py
"""
import torch.nn as nn

class TSYResidualBlock1d(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size=4):
        super(TSYResidualBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1)
        self.batchnorm = nn.BatchNorm1d(num_features=out_channels)
        self.shortcut = self._shortcut(in_channels, out_channels)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        h = self.conv1(x)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.relu2(h)
        h = self.conv3(h)
        h = self.batchnorm(h)
        shortcut = self.shortcut(h)
        return self.relu3(h + shortcut)

    def _shortcut(self, in_channels, out_channels):
        if in_channels != out_channels:
            return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        else:
            lambda x: x