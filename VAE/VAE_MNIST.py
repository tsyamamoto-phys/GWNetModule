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

    def __init__(self, in_features, hidden_features):
        super(MNIST_VAE, self).__init__()
        '''
        hidden_features: the number of the hidden variables
        hidden_params: the number of the parameters the posterior distribution has.
        Assuming the posterior is the isotropic gaussian,
        hidden_params = hidden_features + 1 (1 comes from the variance).
        '''
        self.hidden_features = hidden_features
        self.hidden_params = hidden_features + 1
        
        self.encoder = MNIST_Encoder(in_features, self.hidden_params)
        self.decoder = MNIST_Decoder(hidden_features, in_features)



        
    def _sampling_from_standard_normal(self, seed=False):
        if not seed: torch.manual_seed(seed)
        epsilon = torch.normal(mean=torch.zeros(self.hidden_features))
        return epsilon
        

    def forward(self, x):

        '''
        For now, this is just schematic.
        All following sentences MUST be checked.
        '''

        # Encode the input into the posterior's parameters
        params = self.encoder(x)
        mu = params[:-1]
        sigma = params[-1]

        # Sampling from the standard normal distribution
        # and reparametrize.
        epsilon = self._sampling_from_standard_normal()
        z = mu + sigma * epsilon

        # Decode the hidden variables
        x_tilde = self.decoder(z)
        return(x_tilde)


    
    

if __name__=='__main__':


    vae = MNIST_VAE(288, 10)
    x = vae._sampling_from_standard_normal()
    plt.figure()
    plt.hist(x, bins=100)
    plt.show()




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
