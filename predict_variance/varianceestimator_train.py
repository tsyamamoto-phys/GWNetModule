import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import argparse
import time

import os, sys
sys.path.append(os.environ['HOME']+'/GWNetModule')
from common import noise_inject, mkdir
from MyNetModule import DanielDeeperNet, MeanRelativeError, errorbarError


class ErrorEstimateNet(nn.Module):

    def __init__(self):
        super(ErrorEstimateNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=16)
        self.pool1 = nn.MaxPool1d(4, stride=4)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=16, dilation=2)
        self.pool2 = nn.MaxPool1d(4, stride=4)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=16, dilation=2)
        self.pool3 = nn.MaxPool1d(4, stride=4)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=32, dilation=2)
        self.pool4 = nn.MaxPool1d(4, stride=4)

        self.dense1 = nn.Linear(in_features=512*14, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=64)
        self.dense3 = nn.Linear(in_features=64, out_features=2)



    def forward(self, inputs):
        x = F.relu(self.pool1(self.conv1(inputs)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = F.relu(self.pool4(self.conv4(x)))
        x = x.view(-1, 512*14)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)

        x = torch.exp(x)
        return x

        """
        Sigma = torch.zeros_like(x)
        Sigma[:,0] = x[:,0]**2.0 + x[:,1]**2.0
        Sigma[:,1] = x[:,1]*(x[:,0] + x[:,2])
        Sigma[:,2] = x[:,1]**2.0 + x[:,2]**2.0
        return Sigma
        """


class cdf_error(nn.Module):

    # based on chi-square distribution w\ d.o.f = 2
    
    def __init__(self):
        super(cdf_error, self).__init__()
        self.chi2_cdf = (lambda x: 1-torch.exp(-x/2.0))
'''
    def _sigma2lambda(self, Sigma):
        """
        For l=2,
        Sigma[:,0] = Sigma_11
        Sigma[:,1] = Sigma_12,
        Sigma[:,2] = Sigma_22.
        """

        Lambda = torch.zeros_like(Sigma)
        det = Sigma[:,1]*Sigma[:,1] - Sigma[:,0]*Sigma[:,2]
        
        Lambda[:,0] = Sigma[:,2] / det
        Lambda[:,1] = -Sigma[:,1] / det
        Lambda[:,2] = Sigma[:,0] / det
        return Lambda


    def _normalize_square_sum(self, df, Lambda):
        
        f0f0 = df[:,0] * df[:,0]
        f0f1 = df[:,0] * df[:,1]
        f1f1 = df[:,1] * df[:,1]
        # non diagonal case
        a = f0f0 * Lambda[:,0] + 2 * f0f1 * Lambda[:,1] + f1f1 * Lambda[:,2]
        return a

    def _cdf_error(self, df, Sigma):
        Lambda = self._sigma2lambda(Sigma)
        y = self._normalize_square_sum(df, Lambda)
        y_sort, _ = torch.sort(y)
        cdf_t = self.chi2_cdf(y_sort)

        Nb = df.size()[0]
        cdf_e = torch.arange(0.0, 1.0, 1.0/Nb, out=torch.FloatTensor())
        if torch.cuda.is_available(): cdf_e = cdf_e.cuda()
        error = torch.sum((cdf_t - cdf_e)**2.0) / 2.0
        return error

    def forward(self, preds, Sigma, labels):
        cdf_error = self._cdf_error(preds - labels, Sigma)
        return cdf_error 
        
'''

parser = argparse.ArgumentParser(description="parse args")
#parser.add_argument('-n', '--num-epochs', default=5, type=int)
#parser.add_argument('--datadir', default='/home/', type=str)
parser.add_argument('--savedir', type=str)
#parser.add_argument('--d', type=float)
args = parser.parse_args()


modeldir = '190220/'
savedir = args.savedir
mkdir(savedir)

pred_net = DanielDeeperNet(out_features=2)
pred_net.cuda()
pred_net.load_state_dict(torch.load(modeldir+'check_state.pth'))
for param in pred_net.parameters():
    param.requires_grad = False

var_net = ErrorEstimateNet()
var_net.cuda()

#criterion = errorbarError(d=args.d)
criterion = nn.L1Loss()
optimizer = optim.Adam(var_net.parameters())

datadir = os.environ['HOME']+'/gwdata/'
trainwave = np.load(datadir+'TrainEOB_hPlus.npy')
trainlabel = torch.tensor(np.genfromtxt(datadir+'TrainLabel_new.dat')[:,0:2], dtype=torch.float32)



#snr_list = np.append(np.arange(4.0, 2.0, -0.2),
#                     np.arange(2.0, 0.9, -0.1))
#

snr_list = [3.9]

tik = time.time()
for snr_loc in snr_list:

    pSNR = [snr_loc, 4.0]
    
    if snr_loc>3.5: nepoch = 20
    elif snr_loc>2.0: nepoch = 10
    else: nepoch = 20
    
    for epoch in range(nepoch):

        trainsignal, _ = noise_inject([trainwave], pSNR)
        trainsignal = torch.tensor(trainsignal, dtype=torch.float32)
        traindata = torch.utils.data.TensorDataset(trainsignal,trainlabel)
        train_loader = torch.utils.data.DataLoader(traindata, batch_size=64, shuffle=True, num_workers=4, drop_last=True)

        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):

            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            preds = pred_net(inputs)
            df_t = preds-labels
            sigma2 = var_net(inputs)
            loss = criterion(df_t**2.0, sigma2)

            running_loss += loss.data
            loss.backward()
            optimizer.step()
            
        with open(savedir+'log4check_runningloss_var.txt', 'a') as f:
            f.write("%.1f %.d %.5f \n" % (snr_loc, epoch+1, running_loss/(i+1) ))


torch.save(var_net.state_dict(), savedir+'check_state_varnet.pth')
torch.save(optimizer.state_dict(), savedir+'check_optim_varnet.pth')

tok = time.time()


print("elapsed time:{0} [sec]".format(tok-tik))

