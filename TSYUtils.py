"""
TSYUtils.py
"""

import torch.optim as optim

def _TSYMakeOptimizer(netparameters, name, **kwargs):

    return optim.__dict__[name](netparameters, **kwargs)