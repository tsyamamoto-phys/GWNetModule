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
        self.conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4)
        self.fc1 = nn.Linear(25*25*8, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        output = self.conv(x)
        output = self.fc1(output.view(-1,25*25*8))
        output = F.relu(output)
        output = self.out(output)
        return output

    def model(self, x_data, y_data):
        # define prior distributions
        fc1w_prior = Normal(loc=torch.zeros_like(self.fc1.weight),
                            
