"""
TSYAutoEncoder.py
(Takahiro S. Yamamoto)
"""
import torch
import torch.nn as nn
import json
import _utils as u
#from . import _utils as u


class TSYAutoEncoder(nn.Module):

    def __init__(self, netstructure):
        super(TSYAutoEncoder, self).__init__()

        gl = u.GenerateLayer()

        if isinstance(netstructure, str):
            with open(netstructure, "r") as f:
                print(f"net structure is loaded from `{netstructure}`")
                netstructure = json.load(f)

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
        print("Encode")
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


    def FreezeEncoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False



class TSYVariationalAutoEncoder(nn.Module):

    def __init__(self, netstructure, cudaflg = False):
        super(TSYVariationalAutoEncoder, self).__init__()

        self.cudaflg = cudaflg

        if isinstance(netstructure, str):
            with open(netstructure, "r") as f:
                print(f"net structure is loaded from `{netstructure}`")
                netstructure = json.load(f)


        gl = u.GenerateLayer()

        encoderlayers = []
        for l in netstructure["Encoder"]:
            layername = l["lname"]
            print("encoder: ", layername)
            encoderlayers.append(gl.LayersDict[layername](**(l["params"])))
        self.encoder = nn.ModuleList(encoderlayers)
        # output of Encoder should be divided into a mean and a variance of a Gaussian distribution.
        Nhid = netstructure["Dimension of hidden variable"]
        Nin = self.encoder[-2].out_features  # I need to sophisticate this part.
        self.encoder_mean = nn.Linear(Nin, Nhid)
        self.encoder_logvar = nn.Linear(Nin, Nhid)


        decoderlayers = []
        for l in netstructure["Decoder"]:
            layername = l["lname"]
            print("decoder: ", layername)
            decoderlayers.append(gl.LayersDict[layername](**(l["params"])))
        self.decoder = nn.ModuleList(decoderlayers)


    def encode(self, x):

        for l in self.encoder:
            x = l(x)
        mu = self.encoder_mean(x)
        logvar = self.encoder_logvar(x)
        return mu, logvar


    def decode(self, z):
        for l in self.decoder:
            z = l(z)
        return z
        

    def forward(self, x):
        # encode
        mu, logvar = self.encode(x)

        # sampling
        eps = torch.randn_like(logvar)
        if self.cudaflg: eps = eps.cuda()
        std = logvar.mul(0.5).exp_()
        z = eps.mul(std).add_(mu)

        # decode
        outputs = self.decode(z)

        return outputs, mu, logvar


if __name__ == "__main__":

    from collections import OrderedDict
    from torchinfo import summary
    import json

    params = "testconfig.json"
    net = TSYVariationalAutoEncoder(params)
    print("net is defined.")
    summary(net, inputsize=(1,2,1000))

    # trial
    inputs = torch.empty((10, 2, 1000)).normal_()
    print(f"Input size: {inputs.size()}")
    outputs = net(inputs)
    print(f"Output size: {outputs.size()}")
