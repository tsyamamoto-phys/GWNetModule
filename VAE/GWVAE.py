import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import os, sys
sys.path.append(os.pardir)
from MyNetModule import DanielDeeperNet




class GW_Encoder(nn.Module):

    def __init__(self, in_features, out_features):
        super(GW_Encoder, self).__init__()
        self.dense1 = nn.Linear(in_features=in_features, out_features=1024)
        self.dense2 = nn.Linear(in_features=1024, out_features=256)
        self.dense3 = nn.Linear(in_features=256, out_features=128)        
        self.dense4 = nn.Linear(in_features=128, out_features=out_features)

    
    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = self.dense4(x)
        return x



class GW_Decoder(nn.Module):

    def __init__(self, in_features, out_features):
        super(GW_Decoder, self).__init__()
        self.dense1 = nn.Linear(in_features=in_features, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=256)
        self.dense3 = nn.Linear(in_features=256, out_features=1024)        
        self.dense4 = nn.Linear(in_features=1024, out_features=out_features)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = self.dense4(x)
        return x



class GW_Interpreter(nn.Module):

    def __init__(self, in_features, out_features):
        super(GW_Interpreter, self).__init__()
        self.dense1 = nn.Linear(in_features=in_features, out_features=32)
        self.dense2 = nn.Linear(in_features=32, out_features=32)
        self.dense3 = nn.Linear(in_features=32, out_features=out_features)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x
    



class GW_VAE(nn.Module):

    def __init__(self, in_features, hidden_vars, phys_params, stddiag=True, cudaflg=True):
        super(GW_VAE, self).__init__()
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
        
        self.hidden_vars = hidden_vars
        if stddiag:
            self.stdsize = hidden_vars
        else:
            self.stdsize = int((hidden_vars*(hidden_vars+1))/2)

        self.phys_params = phys_params
        
        self.encoder = GW_Encoder(in_features = in_features,
                                  out_features = self.stdsize + self.hidden_vars)

        self.decoder = GW_Decoder(in_features = self.hidden_vars,
                                  out_features = in_features)
        
        self.interpreter = GW_Interpreter(in_features = self.hidden_vars,
                                          out_features = self.phys_params)




    def encode(self, x):
        x = self.encoder(x)
        return self._vec2mustd(x)

    
    def decode(self, x):
        x = self.decoder(x)
        return x

    def interpret(self, x):
        x = self.interpreter(x)
        return x


    
    
    def _sampling_from_standard_normal(self):
        epsilon = torch.normal(mean=torch.zeros(self.hidden_vars))
        return epsilon



    def _vec2matrix(self, vec):

        nbatch = vec.size()[0]
        mat = torch.zeros((nbatch, self.hidden_vars, self.hidden_vars))

        for n in range(nbatch):
            for i in range(self.hidden_vars):
                for j in range(self.hidden_vars):
                    if i<=j:
                        mat[n, i, j] = vec[n, i*self.hidden_vars - int(i*(i-1)/2) + (j-i)]
                    else:
                        mat[n, i, j] = mat[n, j, i]
        
        return mat

    

    def _vec2mustd(self, vec):
        '''
        Assume the output of the encoder is log(Sigma).
        This method returns the mu and the Sigma.
        '''

        mu = vec[:, 0:self.hidden_vars]

        if self.stddiag:
            sigma2 = torch.exp(vec[:, -self.stdsize:])
            return mu, sigma2
        
        else:
            Sigma = self._vec2matrix(vec[:, -self.stdsize:])
            return mu, Sigma



    
    

    def forward(self, x, mode):

        '''
        For now, only the case for stddiag=True is implemented.
        '''

        assert (mode is 'vae') or (mode is 'inference'), "invalid mode (use 'vae' or 'inference')"

        
        # Encode the input into the posterior's parameters
        mu, Sigma = self.encode(x)
        
        # Sampling from the standard normal distribution
        # and reparametrize.
        epsilon = self._sampling_from_standard_normal()
        if self.cudaflg: epsilon = epsilon.cuda()
        if self.stddiag: z = mu + torch.sqrt(Sigma) * epsilon
        else: z = mu + torch.mm(torch.sqrt(Sigma), epsilon)

        # Decode the hidden variables
        x_tilde = self.decode(z)
        phys_param = self.interpret(z)

        if mode=='vae':
            return x_tilde, mu, Sigma, phys_param
        elif mode=='inference':
            return phys_param



    


class loss_for_vae(nn.Module):

    def __init__(self, beta=1e-3):
        super(loss_for_vae, self).__init__()
        self.beta = beta
        
    def forward(self, x, x_rec, t, t_pre, mu, Sigma):

        '''
        For now, only the case for stddiag=True is implemented.
        You will get the error if stddiag=False.
        '''
        
        KLloss = -(1.0 + torch.log(Sigma) - mu**2.0 - Sigma).sum(dim=-1) / 2.0
        Reconstruction_loss = torch.mean((torch.abs(x_rec - x) ** 2.0) / 2.0, dim=-1)
        Parameter_loss = torch.mean((torch.abs(t_pre - t) ** 2.0) / 2.0, dim=-1)
        total_loss = KLloss + Reconstruction_loss + self.beta * Parameter_loss
        return torch.mean(total_loss)




if __name__ == '__main__':

    datadir = '../testset/'
    trainsignal = torch.Tensor(np.load(datadir+'dampedsinusoid_train.npy'))
    trainlabel = torch.Tensor(np.genfromtxt(datadir+'dampedsinusoid_trainLabel.dat'))

    traindata = torch.utils.data.TensorDataset(trainsignal, trainlabel)
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=512, shuffle=True, num_workers=4)



    vae = GW_VAE(8192, 16, 2, cudaflg=True)
    vae.cuda()
    criterion = loss_for_vae(beta=1e-1)
    optimizer = optim.Adam(vae.parameters())

    

    for epoch in range(100):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs, mu, sigma, predicts = vae(inputs, mode='vae')
            loss = criterion(inputs, outputs, labels, predicts, mu, sigma)
            running_loss += loss.data

            loss.backward()
            optimizer.step()

        print("[ %2d] %.3f" % (epoch, loss))




        
    
    # inference

    x = trainsignal[11:12]
    t = trainlabel[11:12].detach().numpy()

    pre_list = []
    for _ in range(10000):

        with torch.no_grad():
            f_pre = vae(x.cuda(), mode='inference')
            pre_list.append(f_pre.cpu().detach().numpy())

    pre_list = np.array(pre_list)
    print(pre_list[0:10,0,0])

    print(t.shape)


    x_tilde, mu, sigma, f_pre = vae(x.cuda(), mode='vae')
    plt.figure()
    plt.plot(x.detach().numpy()[0])
    plt.plot(x_tilde.cpu().detach().numpy()[0])


    
    plt.figure(figsize=(12,9))
    plt.plot(pre_list[:,0,0], pre_list[:,0,1], '+')
    plt.plot(t[0,0], t[0,1], 'o', color='k')
    plt.xlabel('frequency')
    plt.ylabel('damping frequency')
    plt.show()


