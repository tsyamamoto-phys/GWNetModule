"""
vaenew.py
(Takahiro S. Yamamoto)
"""
import numpy as np
import torch
import torch.nn as nn
from . import _utils as u

class GenerateLayer():
    def __init__(self):
        self.LayersDict = {}
        self.LayersDict["conv1d"] = nn.Conv1d
        self.LayersDict["maxpool1d"] = nn.MaxPool1d
        self.LayersDict["relu"] = nn.ReLU
        self.LayersDict["upsample"] = nn.Upsample
        self.LayersDict["convtranspose1d"] = nn.ConvTranspose1d

    def __call__(self, key):
        return self.LayersDict[key]


class TSYAutoEncoder(nn.Module):

    def __init__(self, netstructure):
        super(TSYAutoEncoder, self).__init__()

        gl = GenerateLayer()

        encoderlayers = []
        for l in netstructure["Encoder"]:
            layername = l["lname"]
            print("encoder: ", layername)
            encoderlayers.append(gl.LayersDict[layername](**(l["params"])))
        self.encoder = nn.ModuleList(encoderlayers)

        decoderlayers = []
        for l in netstructure["Decoder"]:
            layername = l["lname"]
            print("decoder: ", layername)
            decoderlayers.append(gl.LayersDict[layername](**(l["params"])))
        self.decoder = nn.ModuleList(decoderlayers)


    def Encode(self, x):
        print("Encode\n")
        for i, l in enumerate(self.encoder):
            x = l(x)
            print(i, x.size())
        return x


    def Decode(self, x):
        print("Decode")
        for i, l in enumerate(self.decoder):
            x = l(x)
            print(i, x.size())
        return x


if __name__ == "__main__":

    from collections import OrderedDict
    params = []
    params.append({"lname": "conv1d",\
        "params": {"in_channels":1, "out_channels":8, "kernel_size":4}})
    params.append({"lname": "maxpool1d",\
        "params": {"kernel_size":4}})
    params.append({"lname": "relu",\
        "params": {}})


    net = TSYAutoEncoder(params)

    inputs = torch.empty((10,1,512)).uniform_(0.0, 1.0)
    outputs = net.Encode(inputs)
