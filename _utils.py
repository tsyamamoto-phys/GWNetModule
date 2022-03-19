"""
_utils.py
"""
import math
import torch.nn as nn
import gwnet.TSYLayers as TSYLayers

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
        # Convolution
        self.LayersDict["conv1d"] = nn.Conv1d
        self.LayersDict["maxpool1d"] = nn.MaxPool1d
        self.LayersDict["avgpool1d"] = nn.AvgPool1d
        self.LayersDict["conv2d"] = nn.Conv2d
        self.LayersDict["maxpool2d"] = nn.MaxPool2d
        self.LayersDict["avgpool2d"] = nn.AvgPool2d
        # Activation functions
        self.LayersDict["relu"] = nn.ReLU
        self.LayersDict["leaky relu"] = nn.LeakyReLU
        self.LayersDict["negative relu"] = TSYLayers.TSYNegativeReLU
        self.LayersDict["sigmoid"] = nn.Sigmoid
        self.LayersDict["tanh"] = nn.Tanh
        # Deconvolution
        self.LayersDict["upsample"] = nn.Upsample
        self.LayersDict["convtranspose1d"] = nn.ConvTranspose1d
        # Utility
        self.LayersDict["flatten"] = nn.Flatten
        self.LayersDict["Unflatten"] = nn.Unflatten
        # Linear
        self.LayersDict["linear"] = nn.Linear
        # Resnet (implemented by TSY)
        self.LayersDict["residual block"] = TSYLayers.TSYResidualBlock1d

    def __call__(self, key):
        return self.LayersDict[key]
