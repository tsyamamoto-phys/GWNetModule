import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt





class GW_Encoder(nn.Module):

    def __init__(self, in_features, out_features):
        super(GW_Encoder, self).__init__()
        self.dense1 = nn.Linear(in_features=in_features, out_features=512)
        self.dense2 = nn.Linear(in_features=512, out_features=128)
        self.dense3 = nn.Linear(in_features=128, out_features=out_features)

    
    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x



class GW_Decoder(nn.Module):

    def __init__(self, in_features, out_features):
        super(GW_Decoder, self).__init__()
        self.dense1 = nn.Linear(in_features=in_features, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=512)
        self.dense3 = nn.Linear(in_features=512, out_features=out_features)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
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

    def __init__(self, in_features, hidden_vars, phys_params, stddiag=True, cuda=True):
        super(GW_VAE, self).__init__()
        '''
        hidden_features: the number of the hidden variables
        hidden_params: the number of the parameters the posterior distribution has.
        stddiag: If True, the std of hidden variables is a diagonal matrix.
        Assuming the posterior is the isotropic gaussian,
        hidden_params = hidden_features + 1 (1 comes from the variance).
        '''

        self.cuda = cuda
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


    
    
    def _sampling_from_standard_normal(self, seed=False):
        if not seed: torch.manual_seed(seed)
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



    
    

    def forward(self, x):

        '''
        For now, only the case for stddiag=True is implemented.
        '''

        # Encode the input into the posterior's parameters
        mu, Sigma = self.encode(x)
        
        # Sampling from the standard normal distribution
        # and reparametrize.
        epsilon = self._sampling_from_standard_normal()
        if self.cuda: epsilon.cuda()
        if self.stddiag: z = mu + torch.sqrt(Sigma) * epsilon
        else: z = mu + torch.mm(torch.sqrt(Sigma), epsilon)

        # Decode the hidden variables
        x_tilde = self.decode(z)
        phys_param = self.interpret(z)
        
        return(x_tilde, mu, Sigma, phys_param)


        


class loss_for_vae(nn.Module):

    def __init__(self, beta=1e-2):
        super(loss_for_vae, self).__init__()
        self.beta = torch.Tensor(beta)
        
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

    time = np.arange(0.0, 1.0, 1.0/4096)
    freq = 30.0
    x = torch.Tensor(np.sin(2.0*np.pi*freq*time).reshape(1, -1))

    vae = GW_VAE(4096, 10, 1, cuda=False)
    x_tilde, mu, sigma, f_pre = vae(x)
    print(x_tilde)
    print(mu)
    print(sigma)
    print(f_pre)

