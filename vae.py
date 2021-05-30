"""
vae.py
(Takahiro S. Yamamoto)
"""
import numpy as np
import torch
import torch.nn as nn
from . import _utils as u


class TSYAutoEncoder(nn.Module):

    def __init__(self, in_features, params, in_channels=2, printflg=False):
        super(TSYAutoEncoder, self).__init__()

        self.in_features = in_features
        self.in_channels = in_channels

        conv_c = params['conv_channels']
        conv_k = params['conv_kernels']
        pool_k = params['pool_kernels']
        deconv_c = params['deconv_channels']
        deconv_k = params['deconv_kernels']
        deconv_p = params['deconv_pads']
        deconv_op = params['deconv_outpads']
        upsample_s = params['upsample_scales']

        self.Nconv = len(params['conv_channels'])
        self.Npool = len(params['pool_kernels'])
        self.Ndeconv = len(params['deconv_channels'])
        self.Nupsample = len(params['upsample_scales'])
        self.L = 0

        """
        Define layers in encoder.
        """
        encoderlayers = []

        encoderlayers.append(nn.Conv1d(in_channels, out_channels=conv_c[0], kernel_size=conv_k[0]))
        encoderlayers.append(nn.ReLU())
        print("Conv")
        self.L = u._cal_length(in_features, conv_k[0], printflg=printflg)

        if pool_k[0] is not None:
            encoderlayers.append(nn.MaxPool1d(kernel_size=pool_k[0]))
            print("Pooling")
            self.L = u._cal_length(self.L, pool_k[0], pool_k[0], printflg=printflg)

        for i in range(self.Nconv -1):
            encoderlayers.append(nn.Conv1d(in_channels=conv_c[i], out_channels=conv_c[i+1], kernel_size=conv_k[i+1]))
            encoderlayers.append(nn.ReLU())
            print("Conv")
            self.L = u._cal_length(self.L, conv_k[i+1], printflg=printflg)

            if pool_k[i+1] is not None:
                encoderlayers.append(nn.MaxPool1d(kernel_size=pool_k[i+1]))
                print("Pooling")
                self.L = u._cal_length(self.L, pool_k[i+1], pool_k[i+1], printflg=printflg)

        self.encoder = nn.ModuleList(encoderlayers)


        """
        Define layers in decoder.
        """

        decoderlayers = []

        decoderlayers.append(nn.Upsample(scale_factor=upsample_s[0]))
        print("Upsampling")
        self.L = u._cal_length4upsample(self.L, upsample_s[0], printflg=printflg)

        decoderlayers.append(nn.ConvTranspose1d(in_channels=conv_c[-1], out_channels=deconv_c[0], kernel_size=deconv_k[0], padding=deconv_p[0], output_padding=deconv_op[0]))
        decoderlayers.append(nn.ReLU())
        print("ConvTranpose")
        self.L = u._cal_length4deconv(self.L, deconv_k[0], pad=deconv_p[0], outpad=deconv_op[0], printflg=printflg)

        for i in range(self.Ndeconv -1):

            decoderlayers.append(nn.Upsample(scale_factor=upsample_s[i+1]))
            print("Upsampling")
            self.L = u._cal_length4upsample(self.L, upsample_s[i+1], printflg=printflg)

            decoderlayers.append(nn.ConvTranspose1d(in_channels=deconv_c[i], out_channels=deconv_c[i+1], kernel_size=deconv_k[i+1], padding=deconv_p[i+1], output_padding=deconv_op[i+1]))
            print("ConvTranpose")
            self.L = u._cal_length4deconv(self.L, deconv_k[i+1],  pad=deconv_p[i+1], outpad=deconv_op[i+1], printflg=printflg)
            if not (i == self.Ndeconv-2):
                decoderlayers.append(nn.ReLU())


        self.decoder = nn.ModuleList(decoderlayers)

        if self.L == in_features:
            pass
        else:
            print("#"*60)
            print("# CAUTION: input size and output size do not coincide.")
            print("#"*60)

    
    def TSYEncoder(self, x):

        for l in self.encoder:
            x = l(x)
            print(x.size())

        return x

    def TSYDecoder(self, z):

        for l in self.decoder:
            z = l(z)
            print(z.size())

        return z


    def __call__(self, x):
        print("input size: ")
        print(x.size())
        z = self.TSYEncoder(x)
        print("internal size: ")
        print(z.size())
        return self.TSYDecoder(z)


if __name__ == "__main__":

    params = {
        'conv_channels': [8,8,8],
        'conv_kernels': [16,16,16],
        'pool_kernels': [2,2,2],
        'deconv_channels': [8,8,2],
        'deconv_kernels': [16,16,16],
        'deconv_dilations': [1,1,1],
        'deconv_pads': [0,0,0],
        'upsample_scales': [2,2,2]
    }

    net = TSYAutoEncoder(in_features=512, params=params, printflg=True)

    inputs = torch.empty((10,2,512)).uniform_(0.0, 1.0)
    outputs = net(inputs)
    print(outputs.size())