"""
illinois.py

The neural networks that are used in [1] are implemented.

ref
[1] D.George & E.A.Huerta, PRD 97, 044039
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class deepernet(nn.Module):

    def __init__(self, out_features=2):
        super(deepernet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=16)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=16, dilation=2)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=16, dilation=2)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=32, dilation=2)
        self.pool4 = nn.MaxPool1d(4)
        self.dense1 = nn.Linear(in_features=7168, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=64)
        self.dense3 = nn.Linear(in_features=64, out_features=out_features)

    def forward(self, inputs):
        x = F.relu(self.pool1(self.conv1(inputs)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = F.relu(self.pool4(self.conv4(x)))
        x = x.view(-1, 7168)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)

        return x



class shallowernet(nn.Module):

    def __init__(self, out_features=2):
        super(shallowernet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=16)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8, dilation=4)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8, dilation=4)
        self.pool3 = nn.MaxPool1d(4)
        self.dense1 = nn.Linear(in_features=7616, out_features=64)
        self.dense2 = nn.Linear(in_features=64, out_features=out_features)
        
    def forward(self, inputs):
        x = F.relu(self.pool1(self.conv1(inputs)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = x.view(-1,7616)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        
        return x



if __name__=='__main__':

    N = 5
    L = 8192
    x = torch.empty((N,1,L)).normal_(0.0, 1.0).cuda()

    net = shallowernet()
    net.cuda()

    y = net(x)
    print(y.size())
    print(y)
