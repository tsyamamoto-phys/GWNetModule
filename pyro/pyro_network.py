import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

import pyro
from pyro.distributions import Normal
from pyro.distributions import Categorical
from pyro.optim import Adam
from pyro.infer import SVI
from pyro.infer import Trace_ELBO

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()



class BNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(BNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=16)
        self.pool1 = nn.MaxPooling1d(4, stride=4)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8, dilation=4)
        self.pool2 = nn.MaxPooling1d(4, stride=4)
        self.conv3= nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8, dilation=4)
        self.pool3 = nn.MaxPooling1d(4, stride=4)
        self.fc1 = nn.Linear(64*119, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = x.view(-1, 64*119)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def model(self, x_data, y_data):
        # define prior distributions
        fc1w_prior = Normal(loc=torch.zeros_like(self.fc1.weight),
                            scale=torch.ones_like(self.fc1.weight))
        fc1b_prior = Normal(loc=torch.zeros_like(self.fc1.bias),
                            scale=torch.ones_like(self.fc1.bias))
        fc1w_prior = Normal(loc=torch.zeros_like(self.fc1.weight),
                            scale=torch.ones_like(self.fc1.weight))
        fc1w_prior = Normal(loc=torch.zeros_like(self.fc1.weight),
                            scale=torch.ones_like(self.fc1.weight))
