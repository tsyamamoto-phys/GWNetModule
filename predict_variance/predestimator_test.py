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
import MyNetModule as mnm


parser = argparse.ArgumentParser(description="parse args")
#parser.add_argument('-n', '--num-epochs', default=5, type=int)
#parser.add_argument('--datadir', default='/home/', type=str)
parser.add_argument('--modeldir', default='190220/', type=str)
parser.add_argument('--resultdir', type=str)
args = parser.parse_args()


modeldir = args.modeldir
resultdir = args.resultdir
mkdir(resultdir)

net = mnm.DanielDeeperNet(2)
net.load_state_dict(torch.load(modeldir+'check_state_prednet.pth'))

datadir = os.environ['HOME']+'/ringdown_net/'
testwave = np.load(datadir+'TestEOB_hPlus.npy')
labels = np.genfromtxt(datadir+'TestLabel_new.dat')[:,0:2]

snr_list = np.arange(3.9, 0.5, -0.1) 

for pSNR in snr_list:

    with torch.no_grad():
        testsignal, _ = noise_inject([testwave], pSNR)
        testsignal = torch.tensor(testsignal, dtype=torch.float32)

        preds = net(testsignal)
        preds = preds.cpu().detach().numpy()

        with open(resultdir+'pred_outputs_%.1f.txt'%pSNR, 'a+') as f:
            for i in range(preds.shape[0]):
                f.write("%.3f %.3f %.3f %.3f\n" % (labels[i,0], labels[i,1], preds[i,0], preds[i,1]))
