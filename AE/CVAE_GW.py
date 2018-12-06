import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

import math
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import os, sys
sys.path.append(os.pardir)
from common import mkdir



class CVAE_GW(nn.Module):

    def __init__(self, in_features, hidden_features, phys_features, L=1, stddiag=True, cudaflg=True):
        super(CVAE_GW, self).__init__()
        '''
        hidden_features: the number of the hidden variables
        hidden_params: the number of the parameters the posterior distribution has.
        stddiag: If True, the std of hidden variables is a diagonal matrix.
        Assuming the posterior is the isotropic gaussian,
        hidden_params = hidden_features + 1 (1 comes from the variance).
        '''

        self.cudaflg = cudaflg
        self.stddiag = stddiag
        assert self.stddiag==True, "The case std is not diag isn't implemented."

        self.L = L
        
        self.in_features = in_features

        self.hidden_features = hidden_features
        if self.stddiag:
            self.stdsize = self.hidden_features
        else:
            self.stdsize = int((self.hidden_features*(hidden_vars+1))/2)

        self.phys_features = phys_features

        
        # define the layers of encoder
        # In conv layers, you shouldn't use the stride != 1.
        # If stride != 1, it causes the anbiguity of the output size in deconv layers.
        # Use the stride!=1 only for pooling layers.

        def _cal_length(N, k, s=1, d=1):
            return math.floor((N - d*(k-1) -1) / s + 1)

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=16)
        self.c1out = _cal_length(self.in_features, 16)

        self.pool1 = nn.MaxPool1d(4, stride=4)
        self.p1out = _cal_length(self.c1out, 4, s=4)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=16, dilation=2)
        self.c2out = _cal_length(self.p1out, 16, d=2)
        
        self.pool2 = nn.MaxPool1d(4, stride=4)
        self.p2out = _cal_length(self.c2out, 4, s=4)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=16, dilation=2)
        self.c3out = _cal_length(self.p2out, 16, d=2)

        self.pool3 = nn.MaxPool1d(4, stride=4)
        self.p3out = _cal_length(self.c3out, 4, s=4)

        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=32, dilation=2)
        self.c4out = _cal_length(self.p3out, 32, d=2)
        
        self.pool4 = nn.MaxPool1d(4, stride=4)
        self.p4out = _cal_length(self.c4out, 4, s=4)

        self.fc1 = nn.Linear(in_features=512*self.p4out, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=self.hidden_features+self.stdsize)


        # define the layers of decoder
        self.bc1 = nn.Linear(in_features=self.hidden_features, out_features=32)
        self.bc2 = nn.Linear(in_features=32, out_features=self.phys_features)


        
        
    def _sampling_from_standard_normal(self, Nbatch):
        m = Normal(torch.zeros(self.hidden_features), torch.Tensor([1.0]))
        epsilon = m.sample(torch.Size([self.L, Nbatch]))
        return epsilon



    def _vec2matrix(self, vec):

        nbatch = vec.size()[0]
        mat = torch.zeros((nbatch, self.hidden_features, self.hidden_features))

        for n in range(nbatch):
            for i in range(self.hidden_features):
                for j in range(self.hidden_features):
                    if i<=j:
                        mat[n, i, j] = vec[n, i*self.hidden_features - int(i*(i-1)/2) + (j-i)]
                    else:
                        mat[n, i, j] = mat[n, j, i]
        
        return mat


    def _vec2mustd(self, vec):
        '''
        Assume the output of the encoder is log(Sigma).
        This method returns the mu and the Sigma.
        '''

        mu = vec[:, 0:self.hidden_features]

        if self.stddiag:
            sigma2 = torch.exp(vec[:, -self.stdsize:])
            return mu, sigma2
        
        else:
            Sigma = self._vec2matrix(vec[:, -self.stdsize:])
            return mu, Sigma


        

    def encode(self, inputs):
        '''
        For now, only the case for stddiag=True is implemented.
        '''
        
        # Use the layers of encoder
        x = F.relu(self.pool1(self.conv1(inputs)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = F.relu(self.pool4(self.conv4(x)))
        x = x.view(-1, 512*self.p4out)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        mu, sigma = self._vec2mustd(x)

        return mu, sigma
        



    def decode(self, z):

        # Use the layers of decoder and the output size of conv layers.

        z = F.relu(self.bc1(z))
        z = self.bc2(z)

        return z
        

    
    def forward(self, inputs):

        # This is the entire VAE.

        # Encode the input into the posterior's parameters
        Nbatch = inputs.size()[0]
        mu, Sigma = self.encode(inputs)
                
        # Sampling from the standard normal distribution
        # and reparametrize.
        epsilon = self._sampling_from_standard_normal(Nbatch)
        if self.cudaflg: epsilon = epsilon.cuda()
        if self.stddiag: z = mu + torch.sqrt(Sigma).view(-1, Nbatch, self.hidden_features) * epsilon
        else: z = mu + torch.mm(torch.sqrt(Sigma), epsilon)

        # Decode the hidden variables
        y_pred_array = self.decode(z)
        return y_pred_array, mu, Sigma
        
        #y_pred = torch.mean(self.decode(z), dim=0)
        #return y_pred, mu, Sigma
        
        '''
        if mode=='vae':
            return x_tilde, mu, Sigma, phys_param
        elif mode=='inference':
            phys_param = self.interpret(z)
            return phys_param
        '''


    def inference(self, inputs, Nsample=1000):

        outlist = []
        for _ in range(Nsample):

            y, mu, sigma = self.forward(inputs)
            
            if self.cudaflg:
                y = y.cpu()
                
            outlist.append(y.detach().numpy())

        return outlist




class loss_for_vae(nn.Module):

    def __init__(self, alpha=1.0):
        super(loss_for_vae, self).__init__()
        self.alpha = alpha
        
    def forward(self, y_true, y_pred, mu, Sigma):

        '''
        For now, only the case for stddiag=True is implemented.
        You will get the error if stddiag=False.
        '''
        
        KLloss = -(1.0 + torch.log(Sigma) - mu**2.0 - Sigma).sum(dim=-1) / 2.0
        Reconstruction_loss = torch.sum((torch.abs(y_pred - y_true) / y_true), dim=-1)
        total_loss = self.alpha * KLloss + Reconstruction_loss
        return torch.mean(total_loss)



class loss_with_array(nn.Module):
    def __init__(self, alpha=1.0, alter=False):
        super(loss_with_array, self).__init__()
        self.alpha = alpha
        self.alter = alter

    def forward(self, y_true, y_pred_array, mu, Sigma):

        """ y_pred_array has size (L, N, D) """
        N = y_pred_array.size(1)
        ymean = torch.mean(y_pred_array, dim=0)
        ycov = cov_3d(y_pred_array)
        if self.alter:
            KLloss = torch.mean(-(1.0 + torch.log(Sigma) - mu**2.0 - Sigma).sum(dim=-1) / 2.0)
            Rec_loss = 0.0
            for n in range(N):
                m = ymean[n] - y_true[n]
                C = ycov[n]
                Cinv = torch.inverse(C)
                recloss = (torch.log(torch.abs(torch.det(C))) + torch.dot(torch.mv(Cinv,m), m))/2.0
                Rec_loss += recloss.data
            Rec_loss = Rec_loss / N
            
            return KLloss + Rec_loss, KLloss, Rec_loss

        else:
            KLloss = -(1.0 + torch.log(Sigma) - mu**2.0 - Sigma).sum(dim=-1) / 2.0
            Reconstruction_loss = torch.sum((torch.abs(ymean - y_true) / y_true), dim=-1)

            total_loss = self.alpha * KLloss + Reconstruction_loss
            
            return torch.mean(total_loss), torch.mean(KLloss), torch.mean(Reconstruction_loss)
    

def cov(m, y=None):
    if y is not None:
        m = torch.cat((m, y), dim=0)
    m_exp = torch.mean(m, dim=0)
    x = m - m_exp[None,:]
    cov = 1 / (x.size(0)-1) * (x.t()).mm(x)
    return cov

def cov_3d(M, y=None):
    L, N, D = M.size()
    COV = torch.zeros(N,D,D).cuda()
    for n in range(N):
        x = cov(M[:,n,:].view(L,D), y=y)
        COV[n] = x

    return COV



if __name__ == '__main__':


    #filedir = '/home/tap/errorbar/testset/'
    filedir = '/home/tyamamoto/errornet/testset/'

    trainwave = torch.Tensor(np.load(filedir+'dampedsinusoid_train.npy').reshape(-1, 1, 8192))
    trainlabel = torch.Tensor(np.genfromtxt(filedir+'dampedsinusoid_trainLabel.dat'))

    cvae = CVAE_GW(in_features=8192,
                   hidden_features=10,
                   phys_features=2,
                   L=5,
                   cudaflg=False)


    output = cvae(trainwave[0:3].view(-1,1,8192))
    print(output[0].size())


    criterion = loss_with_array()
    loss = criterion(trainlabel[0:3], *output)
    print(loss.size())

