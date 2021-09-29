"""
TSYPlainNetwork.py
"""
import torch
import torch.nn as nn
from . import _utils as u

class TSYPlainNetwork(nn.Module):

    def __init__(self, netstructure, showsize=False):
        super(TSYPlainNetwork, self).__init__()

        gl = u.GenerateLayer()
        layers = []
        for l in netstructure["Net"]:
            layername = l["lname"]
            layers.append(gl.LayersDict[layername](**(l["params"])))
        self.layers = nn.ModuleList(layers)

        self.showsize = showsize

    def forward(self, x):
        for l in self.layers:
            if self.showsize:
                print("layer size: ", x.size())
            x = l(x)
        if self.showsize:
            print("output size: ", x.size())
            self.showsize = False
        return x

    def check_intermediate_outputs(self, x):
        intermediateoutputs = []
        for l in self.layers:
            x = l(x)
            intermediateoutputs.append(x)
        return intermediateoutputs