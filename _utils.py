"""
_utils.py
"""
import math
import torch.nn as nn
from gwnet.TSYLayers import TSYResidualBlock1d

def _cal_length(N, k, s=1, d=1, printflg=False):
    
    ret = math.floor((N - d*(k-1) -1) / s + 1)
    if printflg: print("output length: ", ret)
    return ret


def _cal_length4deconv(N, k, s=1, pad=0, outpad=0, d=1, printflg=False):

    ret = (N-1) * s - 2 * pad + d * (k-1) + outpad + 1
    if printflg: print("output length: ", ret)
    return ret

def _cal_length4upsample(N, scale, printflg=False):

    ret = N * scale
    if printflg: print("output length: ", ret)
    return ret


class GenerateLayer():
    def __init__(self):
        self.LayersDict = {}
        self.LayersDict["conv1d"] = nn.Conv1d
        self.LayersDict["maxpool1d"] = nn.MaxPool1d
        self.LayersDict["avgpool1d"] = nn.AvgPool1d
        self.LayersDict["conv2d"] = nn.Conv2d
        self.LayersDict["maxpool2d"] = nn.MaxPool2d
        self.LayersDict["avgpool2d"] = nn.AvgPool2d
        self.LayersDict["relu"] = nn.ReLU
        self.LayersDict["leaky relu"] = nn.LeakyReLU
        self.LayersDict["upsample"] = nn.Upsample
        self.LayersDict["convtranspose1d"] = nn.ConvTranspose1d
        self.LayersDict["flatten"] = nn.Flatten
        self.LayersDict["linear"] = nn.Linear
        self.LayersDict["Unflatten"] = nn.Unflatten
        self.LayersDict["residual block"] = TSYResidualBlock1d

    def __call__(self, key):
        return self.LayersDict[key]
