import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import numpy as np
import matplotlib.pyplot as plt



class CAE_MNIST(nn.Module):

    def __init__(self, in_features, hidden_features):
        super(CAE_MNIST, self).__init__()
                

        self.in_features = in_features
        self.hidden_features = hidden_features

        
        # define the layers of encoder
        # In conv layers, you shouldn't use the stride != 1.
        # If stride != 1, it causes the anbiguity of the output size in deconv layers.
        # Use the stride!=1 only for pooling layers.

        def _cal_length(N, k, s=1, d=1):
            return math.floor((N - d*(k-1) -1) / s + 1)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4)
        self.c1out = _cal_length(self.in_features, 4)

        self.pool1 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.p1out = _cal_length(self.c1out, 2, s=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4)
        self.c2out = _cal_length(self.p1out, 4)
        
        self.pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.p2out = _cal_length(self.c2out, 2, s=2)
        
        self.fc1 = nn.Linear(in_features=32*self.p2out*self.p2out, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=self.hidden_features)


        # define the layers of decoder
        self.bc1 = nn.Linear(in_features=self.hidden_features, out_features=256)
        self.bc2 = nn.Linear(in_features=256, out_features=32*self.p2out*self.p2out)

        self.upool1 = nn.MaxUnpool2d(2, stride=2)
        self.dconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4)
        self.upool2 = nn.MaxUnpool2d(2, stride=2)
        self.dconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=4)




    def encode(self, inputs, mode='encode'):

        assert (mode=='encode')or(mode=='cae'), "invalide mode; use 'encode' or 'cae'"
        

        # Use the layers of encoder
        x, idx1 = self.pool1(self.conv1(inputs))
        x = F.relu(x)

        x, idx2 = self.pool2(self.conv2(x))
        x = F.relu(x)
        
        x = x.view(-1, 32*self.p2out*self.p2out)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        if mode=='encode':
            return x
        elif mode=='cae':
            return x, idx1, idx2
        else:
            return None
        



    def decode(self, inputs):

        # Use the layers of decoder and the output size of conv layers.
        if isinstance(inputs, tuple):
            z, indices = inputs[0], inputs[1:]
        elif isinstance(inputs, torch.Tensor):
            z = inputs

        N = z.size()[0]

        z = F.relu(self.bc1(z))
        z = F.relu(self.bc2(z))

        z = z.view(N, 32, self.p2out, self.p2out)

        z = self.upool1(z, indices[-1], output_size=(N, 32, self.c2out, self.c2out))
        z = F.relu(self.dconv1(z))
        z = self.upool2(z, indices[-2], output_size=(N, 16, self.c1out, self.c1out))
        z = torch.tanh(self.dconv2(z))

        return z
        

    
    def forward(self, inputs):

        # This is the entire VAE.
        z = self.encode(inputs, mode='cae')
        outputs = self.decode(z)
        return outputs






class loss_for_vae(nn.Module):

    def __init__(self):
        super(loss_for_vae, self).__init__()
        
    def forward(self, x, x_rec, mu, Sigma):

        '''
        For now, only the case for stddiag=True is implemented.
        You will get the error if stddiag=False.
        '''
        
        KLloss = -(1.0 + torch.log(Sigma) - mu**2.0 - Sigma).sum(dim=-1) / 2.0
        Reconstruction_loss = torch.mean((torch.abs(x_rec - x) ** 2.0) / 2.0, dim=-1)
        total_loss = KLloss + Reconstruction_loss
        return torch.mean(total_loss)






if __name__ == '__main__':



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


    imagenet_data_test = torchvision.datasets.MNIST('~/data',
                                               train=False,
                                               download=False,
                                               transform=transform)
    

    testdata_loader = torch.utils.data.DataLoader(imagenet_data_test,
                                                  batch_size=32,
                                                  shuffle=True,
                                                  num_workers=1)




    cae = CAE_MNIST(28, 16)
    cae.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(cae.parameters())


    
    for epoch in range(50):

        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):

            inputs, _ = data
            inputs = inputs.cuda()
            
            optimizer.zero_grad()
            outputs = cae(inputs)
            loss = criterion(inputs, outputs)
            running_loss += loss.data

            loss.backward()
            optimizer.step()    # Does the update

        print("[%2d] %.3f" % (epoch, running_loss / (i+1)))


    with torch.no_grad():
        for _ in range(3):

            j = np.random.randint(100)
            
            inputs, labels = imagenet_data_test[j]
            inputs = inputs.view(1, 1, 28, 28).cuda()
            outputs = cae(inputs)

            inputs = inputs.cpu()
            outputs = outputs.cpu()

            loss = torch.sum(torch.abs(inputs - outputs)**2) / 2.0
            print(loss)
            

            plt.figure()
            plt.imshow(inputs.detach().numpy()[0,0])
            plt.title('original image: %d' % labels)
            
            plt.figure()
            plt.imshow(outputs.detach().numpy()[0,0])
            plt.title('reproduced image: %d' % labels)
        
        plt.show()


