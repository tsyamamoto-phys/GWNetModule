"""
TSYOptimizer.py
"""

import torch.optim as optim

def _TSYMakeOptimizer(netparameters, name, **kwargs):

    if name=="AdaBound":
        import adabound
        return adabound.AdaBound(netparameters, **kwargs)
    else:
        return optim.__dict__[name](netparameters, **kwargs)
