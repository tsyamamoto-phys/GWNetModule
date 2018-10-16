import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt



class MNIST_Encoder(nn.Module):

    def __init__(self, in_features, out_features):
        super(MNIST_Encoder, self).__init__()
        self.dense1 = nn.Linear(in_features=in_features, out_features=512)
        self.dense2 = nn.Linear(in_features=512, out_features=128)
        self.dense3 = nn.Linear(in_features=128, out_features=out_features)

    
    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x



class MNIST_Decoder(nn.Module):

    def __init__(self, in_features, out_features):
        super(MNIST_Decoder, self).__init__()
        self.dense1 = nn.Linear(in_features=in_features, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=512)
        self.dense3 = nn.Linear(in_features=512, out_features=out_features)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = torch.tanh(self.dense3(x))
        return x



class MNIST_AE(nn.Module):

    def __init__(self, in_features, hidden_features):
        super(MNIST_AE, self).__init__()
        self.encoder = MNIST_Encoder(in_features, hidden_features)
        self.decoder = MNIST_Decoder(hidden_features, in_features)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x




class MNIST_VAE(nn.Module):

    def __init__(self, in_features, hidden_vars, stddiag=True):
        super(MNIST_VAE, self).__init__()
        '''
        hidden_features: the number of the hidden variables
        hidden_params: the number of the parameters the posterior distribution has.
        stddiag: If True, the std of hidden variables is a diagonal matrix.
        Assuming the posterior is the isotropic gaussian,
        hidden_params = hidden_features + 1 (1 comes from the variance).
        '''

        self.stddiag = stddiag
        assert self.stddiag==True, "The case std is not diag isn't implemented."
        
        self.hidden_vars = hidden_vars
        if stddiag:
            self.stdsize = hidden_vars
        else:
            self.stdsize = int((hidden_vars*(hidden_vars+1))/2)
        
        self.encoder = MNIST_Encoder(in_features = in_features,
                                     out_features = self.stdsize + self.hidden_vars)
        self.decoder = MNIST_Decoder(in_features = self.hidden_vars,
                                     out_features = in_features)



    def encode(self, x):
        x = self.encoder(x)
        return x

    
    def decode(self, x):
        x = self.decoder(x)
        return x

    
    
    def _sampling_from_standard_normal(self, seed=False):
        if not seed: torch.manual_seed(seed)
        epsilon = torch.normal(mean=torch.zeros(self.hidden_vars))
        return epsilon
        

    def forward(self, x):

        '''
        For now, only the case for stddiag=True is implemented.
        '''

        # Encode the input into the posterior's parameters
        params = self.encoder(x)
        mu, Sigma = self._vec2mustd(params)
        #print("mu=", mu)
        #print("Sigma=", Sigma)
        #print("The dimension of the mu:", mu.size())
        #print("The dimension of the sigma:", Sigma.size())

        # Sampling from the standard normal distribution
        # and reparametrize.
        epsilon = self._sampling_from_standard_normal()
        #print("The dimension of the epsilon:", epsilon.size())
        #print("epsilon=", epsilon)
        
        if self.stddiag: z = mu + torch.sqrt(Sigma) * epsilon
        else: z = mu + torch.mm(torch.sqrt(Sigma), epsilon)
        #print("The dimension of the z:", z.size())
        #print("z=", z)

        # Decode the hidden variables
        x_tilde = self.decoder(z)
        #print("The dimension of the x_tilde:", x_tilde.size())
        
        return(x_tilde, mu, Sigma)





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




class loss_for_vae(nn.Module):

    def __init__(self):
        super(loss_for_vae, self).__init__()
        
    def forward(self, x, x_rec, mu, Sigma):

        '''
        For now, only the case for stddiag=True is implemented.
        You will get the error if stddiag=False.
        '''
        
        KLloss = -(1.0 + torch.log(Sigma) - mu**2.0 - Sigma).sum(dim=-1) / 2.0
        Reconstruction_loss = torch.abs(x_rec - x) ** 2.0
        total_loss = KLloss + Reconstruction_loss
        return total_loss



        




if __name__=='__main__':


    vae = MNIST_VAE(in_features=288, hidden_vars=3)
    
    x = torch.Tensor(np.random.randn(4,288))
    z = vae.encode(x)

    print("The dimension of the hidden variables:", z.size())

    print("Forward calculation started")


    x_tilde = vae.forward(x)
    
    print("Forward calculation finished")




    '''

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, ), (0.5, ))])

    
    imagenet_data = torchvision.datasets.MNIST('~/data',
                                               train=True,
                                               download=True,
                                               transform=transform)
    
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=128,
                                              shuffle=True,
                                              num_workers=4)


    imagenet_data = torchvision.datasets.MNIST('~/data',
                                               train=False,
                                               download=False,
                                               transform=transform)
    

    testdata_loader = torch.utils.data.DataLoader(imagenet_data,
                                                  batch_size=32,
                                                  shuffle=True,
                                                  num_workers=1)




    ae = MNIST_AE(28*28, 15)
    ae = ae.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters())


    for epoch in range(20):
        for i, data in enumerate(data_loader, 0):

            inputs, _ = data
            inputs = inputs.cuda()
            inputs = inputs.view(-1, 28*28)
            
            optimizer.zero_grad()
            outputs = ae(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()    # Does the update



    with torch.no_grad():
        for j in range(3):
            inputs, labels = imagenet_data[j]
            inputs = inputs.view(-1, 28*28).cuda()
            outputs = ae(inputs)
        
            inputs = inputs.cpu().view(-1, 28, 28)
            outputs = outputs.cpu().view(-1, 28, 28)

            plt.figure()
            plt.imshow(inputs[0])
            plt.title('original image: %d' % labels)
            
            plt.figure()
            plt.imshow(outputs[0])
            plt.title('reproduced image: %d' % labels)
        
        plt.show()
'''
