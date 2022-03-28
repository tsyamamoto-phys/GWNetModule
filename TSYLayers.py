"""
TSYLayers.py
"""
import torch.nn as nn
import math

class TSYResidualBlock1d(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size=4, stride=2, padding_adjuster=1):
        super(TSYResidualBlock1d, self).__init__()

        padding = math.floor((kernel_size - 1)/2) + padding_adjuster

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1)
        self.batchnorm = nn.BatchNorm1d(num_features=out_channels)
        self.shortcut = self._shortcut(in_channels, out_channels, stride)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        h = self.conv1(x)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.relu2(h)
        h = self.conv3(h)
        h = self.batchnorm(h)
        shortcut = self.shortcut(x)
        return self.relu3(h + shortcut)

    def _shortcut(self, in_channels, out_channels, stride):
        return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)

    
class TSYNegativeReLU(nn.Module):
    
    def __init__(self):
        super(TSYNegativeReLU, self).__init__()
        self.relu = nn.ReLU()
        
    def forward(self,x):
        return - self.relu(x)

