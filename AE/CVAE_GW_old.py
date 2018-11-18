import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import numpy as np
import matplotlib.pyplot as plt



class CVAE_GW(nn.Module):

    def __init__(self, in_features, hidden_features, phys_features, stddiag=True, cudaflg=True):
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

        self.pool1 = nn.MaxPool1d(4, stride=4, return_indices=True)
        self.p1out = _cal_length(self.c1out, 4, s=4)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=16, dilation=2)
        self.c2out = _cal_length(self.p1out, 16, d=2)
        
        self.pool2 = nn.MaxPool1d(4, stride=4, return_indices=True)
        self.p2out = _cal_length(self.c2out, 4, s=4)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=16, dilation=2)
        self.c3out = _cal_length(self.p2out, 16, d=2)

        self.pool3 = nn.MaxPool1d(4, stride=4, return_indices=True)
        self.p3out = _cal_length(self.c3out, 4, s=4)

        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=32, dilation=2)
        self.c4out = _cal_length(self.p3out, 32, d=2)
        
        self.pool4 = nn.MaxPool1d(4, stride=4, return_indices=True)
        self.p4out = _cal_length(self.c4out, 4, s=4)

        self.fc1 = nn.Linear(in_features=512*self.p4out, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=self.hidden_features+self.stdsize)


        # define the layers of decoder
        self.bc1 = nn.Linear(in_features=self.hidden_features, out_features=128)
        self.bc2 = nn.Linear(in_features=128, out_features=512*self.p4out)

        self.upool1 = nn.MaxUnpool1d(4, stride=4)
        self.dconv1 = nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=32, dilation=2)
        self.upool2 = nn.MaxUnpool1d(4,stride=4)
        self.dconv2 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=16, dilation=2)
        self.upool3 = nn.MaxUnpool1d(4, stride=4)
        self.dconv3 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=16, dilation=2)
        self.upool4 = nn.MaxUnpool1d(4, stride=4)
        self.dconv4 = nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size=16)



        
    def _sampling_from_standard_normal(self):
        epsilon = torch.normal(mean=torch.zeros(self.hidden_features))
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


        

    def encode(self, inputs, mode='encode'):
        '''
        For now, only the case for stddiag=True is implemented.
        '''
        
        assert (mode=='encode')or(mode=='cae'), "invalide mode; use 'encode' or 'cae'"
        

        # Use the layers of encoder
        indices = []
        
        x, idx = self.pool1(self.conv1(inputs))
        x = F.relu(x)
        indices.append(idx)

        x, idx = self.pool2(self.conv2(x))
        x = F.relu(x)
        indices.append(idx)

        x, idx = self.pool3(self.conv3(x))
        x = F.relu(x)
        indices.append(idx)

        x, idx = self.pool4(self.conv4(x))
        x = F.relu(x)
        indices.append(idx)
        
        x = x.view(-1, 512*self.p4out)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        mu, sigma = self._vec2mustd(x)
        
        if mode=='encode':
            return mu, sigma
        elif mode=='cae':
            return mu, sigma, indices
        else:
            return None
        



    def decode(self, z, indices):

        # Use the layers of decoder and the output size of conv layers.

        N = z.size()[0]

        z = F.relu(self.bc1(z))
        z = F.relu(self.bc2(z))

        z = z.view(N, 512, self.p4out)

        z = self.upool1(z, indices[-1], output_size=(N, 256, self.c4out))
        z = F.relu(self.dconv1(z))
        z = self.upool2(z, indices[-2], output_size=(N, 128, self.c3out))
        z = F.relu(self.dconv2(z))
        z = self.upool3(z, indices[-3], output_size=(N, 64, self.c2out))
        z = F.relu(self.dconv3(z))
        z = self.upool4(z, indices[-4], output_size=(N, 32, self.c1out))
        z = self.dconv4(z)

        return z
        

    
    def forward(self, inputs):

        # This is the entire VAE.

        # Encode the input into the posterior's parameters
        mu, Sigma, indices = self.encode(inputs, mode='cae')
                
        # Sampling from the standard normal distribution
        # and reparametrize.
        epsilon = self._sampling_from_standard_normal()
        if self.cudaflg: epsilon = epsilon.cuda()
        if self.stddiag: z = mu + torch.sqrt(Sigma) * epsilon
        else: z = mu + torch.mm(torch.sqrt(Sigma), epsilon)

        # Decode the hidden variables
        x_tilde = self.decode(z, indices)

        return x_tilde, mu, Sigma
        
        '''
        if mode=='vae':
            return x_tilde, mu, Sigma, phys_param
        elif mode=='inference':
            phys_param = self.interpret(z)
            return phys_param
        '''

        

class loss_for_vae(nn.Module):

    def __init__(self, alpha=1.0):
        super(loss_for_vae, self).__init__()
        self.alpha = alpha
        
    def forward(self, x, x_tilde, mu, Sigma):

        '''
        For now, only the case for stddiag=True is implemented.
        You will get the error if stddiag=False.
        '''
        
        KLloss = -(1.0 + torch.log(Sigma) - mu**2.0 - Sigma).sum(dim=-1) / 2.0
        Reconstruction_loss = torch.mean((torch.abs(x_tilde - x) ** 2.0) / 2.0, dim=-1)
        total_loss = self.alpha * KLloss + Reconstruction_loss
        return torch.mean(total_loss)



if __name__ == '__main__':


    filedir = '/home/tyamamoto/testset/'

    trainwave = torch.Tensor(np.load(filedir+'dampedsinusoid_train.npy').reshape(-1, 1, 8192))
    trainlabel = torch.Tensor(np.genfromtxt(filedir+'dampedsinusoid_trainLabel.dat'))
    traindata = torch.utils.data.TensorDataset(trainwave, trainlabel)
    data_loader = torch.utils.data.DataLoader(traindata, batch_size=256, shuffle=True, num_workers=4)

    cvae = CVAE_GW(in_features=8192,
                   hidden_features=2,
                   phys_features=2)
    cvae.cuda()


    criterion = loss_for_vae()
    optimizer = optim.Adam(cvae.parameters())

    modeldir = '/home/tyamamoto/CVAEmodel/'


    cvae.state_dict(torch.load(modeldir+'181024model_50.pt'))

    '''
    for epoch in range(50):

        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):

            inputs, _ = data
            inputs = inputs.cuda()
            
            optimizer.zero_grad()
            outputs, mu, Sigma = cvae(inputs)
            loss = criterion(inputs, outputs, mu, Sigma)
            running_loss += loss.data

            loss.backward()
            optimizer.step()    # Does the update

        print("[%2d] %.5f" % (epoch+1, running_loss / (i+1)))

        if epoch%10==9: torch.save(cvae.state_dict(), modeldir+'181024model_%d.pt'%(epoch+1))

    '''
        

    with torch.no_grad():
        for j in [0, 8140, 16270]:
            
            inputs, labels = traindata[j]
            inputs = inputs.view(1, 1, 8192)
            inputs = inputs.cuda()
            outputs, mu, Sigma = cvae(inputs)

            inputs = inputs.cpu()
            outputs = outputs.cpu()


            plt.figure()
            plt.plot(inputs.detach().numpy()[0,0], label='original signal')
            plt.plot(outputs.detach().numpy()[0,0], label='reproduced signal')
        
        plt.show()
