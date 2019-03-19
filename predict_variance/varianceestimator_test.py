import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import argparse

import os, sys
sys.path.append(os.environ['HOME']+'/GWNetModule')
from common import noise_inject, mkdir
from errornet_module import ErrorEstimateNet_double, log_likelihood_error


parser = argparse.ArgumentParser(description="parse args")
#parser.add_argument('-n', '--num-epochs', default=5, type=int)
#parser.add_argument('--datadir', default='/home/', type=str)
parser.add_argument('--resultdir', type=str)
args = parser.parse_args()


modeldir = '190220/'
resultdir = args.resultdir

net = ErrorEstimateNet_double()
net.load_state_dict(torch.load(resultdir+'check_state_varnet.pth'))

'''
model_dict = net.state_dict()
pretrained_dict = torch.load(modeldir+'check_state.pth')
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
net.load_state_dict(model_dict)
'''

datadir = os.environ['HOME']+'/ringdown_net/'
testwave = np.load(datadir+'TestEOB_hPlus.npy')
labels = np.genfromtxt(datadir+'TestLabel_new.dat')[:,0:2]

snr_list = [4.0] 

for pSNR in snr_list:

    with torch.no_grad():
        testsignal, _ = noise_inject([testwave], pSNR)
        testsignal = torch.tensor(testsignal, dtype=torch.float32)

        preds = net.point_pred(testsignal)
        (var, cov) = net.variance(testsignal)
        preds = preds.cpu().detach().numpy()
        var = var.cpu().detach().numpy()
        cov = cov.cpu().detach().numpy()

        with open(resultdir+'test_outputs_%.1f.txt'%pSNR, 'a+') as f:
            for i in range(preds.shape[0]):
                f.write("%.3f %.3f %.3f %.3f %.6f %.6f %.6f\n" % (labels[i,0], labels[i,1], preds[i,0], preds[i,1], var[i,0], var[i,1], cov[i]))


