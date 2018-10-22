import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt








layer1 = nn.ConvTranspose1d(128, 32, 1)
x = torch.Tensor(np.random.randn(1,128,16))
y = layer1(x)

print(y.size())

