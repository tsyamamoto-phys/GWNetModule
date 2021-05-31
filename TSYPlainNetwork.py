"""
TSYPlainNetwork.py
"""
import torch
import torch.nn as nn
from . import _utils as u

class TSYPlainNetwork(nn.Module):

    def __init__(self, netstructure):
        super(TSYPlainNetwork, self).__init__()

        gl = u.GenerateLayer()
        layers = []
        for l in netstructure["Net"]:
            layername = l["lname"]
            layers.append(gl.LayersDict[layername](**(l["params"])))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x
