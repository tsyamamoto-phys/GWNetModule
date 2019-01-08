import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import numpy as np
#import matplotlib.pyplot as plt

import os, sys
sys.path.append(os.pardir)
from common import noise_inject



class CAE_GW(nn.Module):

    def __init__(self, in_features, hidden_features):
        super(CAE_GW, self).__init__()
                

        self.in_features = in_features
        self.hidden_features = hidden_features

        
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
        self.fc2 = nn.Linear(in_features=128, out_features=self.hidden_features)


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



    def encode(self, inputs):

        # Use the layers of encoder
        self.indices = []
        
        x, idx = self.pool1(self.conv1(inputs))
        x = F.relu(x)
        self.indices.append(idx)

        x, idx = self.pool2(self.conv2(x))
        x = F.relu(x)
        self.indices.append(idx)

        x, idx = self.pool3(self.conv3(x))
        x = F.relu(x)
        self.indices.append(idx)

        x, idx = self.pool4(self.conv4(x))
        x = F.relu(x)
        self.indices.append(idx)
        
        x = x.view(-1, 512*self.p4out)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

        



    def decode(self, z):

        # Use the layers of decoder and the output size of conv layers.

        N = z.size()[0]
        
        z = F.relu(self.bc1(z))
        z = F.relu(self.bc2(z))

        z = z.view(N, 512, self.p4out)

        z = self.upool1(z, self.indices[-1], output_size=[N, 256, self.c4out])
        z = F.relu(self.dconv1(z))
        z = self.upool2(z, self.indices[-2], output_size=[N, 128, self.c3out])
        z = F.relu(self.dconv2(z))
        z = self.upool3(z, self.indices[-3], output_size=[N, 64, self.c2out])
        z = F.relu(self.dconv3(z))
        z = self.upool4(z, self.indices[-4], output_size=[N, 32, self.c1out])
        z = self.dconv4(z)

        return z
        

    
    def forward(self, inputs):

        # This is the entire VAE.
        z = self.encode(inputs)
        outputs = self.decode(z)
        pred = z[:,0:2]
        return outputs, pred



    def inference_for_single_event(self, inputs, Nsample=1000):
        """
        inputs has shape (1, 1, Length) because noise_inject will be carried out before this method
        """
        pass



    

    def inference(self, inputs, Nsample=1000):
        """
        inputs has shape (Nbatch, Length), and type torch.Tensor
        return has shape (Nsample, Nbatch, Length)
        """

        Nb, _, L = inputs.size()
        output = torch.zeros((Nsample, Nb, 2))
        if torch.cuda.is_available():
            output = output.cuda()
        for n in range(Nsample):
            z = self.encode(inputs)
            output[n, :, :] = z[:,0:2]
        return output








class CAE_GW_190104(nn.Module):

    def __init__(self, in_features, hidden_features):
        super(CAE_GW_190104, self).__init__()
                

        self.in_features = in_features
        self.hidden_features = hidden_features

        
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



        # estimation layers
        self.fc1 = nn.Linear(in_features=512*self.p4out, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=2)

        

        # define the layers of decoder
        self.upool1 = nn.MaxUnpool1d(4, stride=4)
        self.dconv1 = nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=32, dilation=2)

        self.upool2 = nn.MaxUnpool1d(4,stride=4)
        self.dconv2 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=16, dilation=2)

        self.upool3 = nn.MaxUnpool1d(4, stride=4)
        self.dconv3 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=16, dilation=2)

        self.upool4 = nn.MaxUnpool1d(4, stride=4)
        self.dconv4 = nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size=16)



    def encode(self, inputs):

        # Use the layers of encoder
        self.indices = []
        
        x, idx = self.pool1(self.conv1(inputs))
        x = F.relu(x)
        self.indices.append(idx)

        x, idx = self.pool2(self.conv2(x))
        x = F.relu(x)
        self.indices.append(idx)

        x, idx = self.pool3(self.conv3(x))
        x = F.relu(x)
        self.indices.append(idx)

        x, idx = self.pool4(self.conv4(x))
        x = F.relu(x)
        self.indices.append(idx)
        
        x = x.view(-1, 512*self.p4out)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

        



    def decode(self, z):

        # Use the layers of decoder and the output size of conv layers.

        N = z.size()[0]
        
        z = self.upool1(z, self.indices[-1], output_size=[N, 256, self.c4out])
        z = F.relu(self.dconv1(z))
        z = self.upool2(z, self.indices[-2], output_size=[N, 128, self.c3out])
        z = F.relu(self.dconv2(z))
        z = self.upool3(z, self.indices[-3], output_size=[N, 64, self.c2out])
        z = F.relu(self.dconv3(z))
        z = self.upool4(z, self.indices[-4], output_size=[N, 32, self.c1out])
        z = self.dconv4(z)

        return z


    def inference(self, inputs):

        z = inputs.view(-1, 512*self.p4out)
        z = F.relu(self.fc1(z))
        z = self.fc2(z)

        return z

    

    
    def forward(self, inputs):

        # This is the entire VAE.
        z = self.encode(inputs)
        outputs = self.decode(z)
        pred = z[:,0:2]
        return outputs, pred



    def inference_for_single_event(self, inputs, Nsample=1000):
        """
        inputs has shape (1, 1, Length) because noise_inject will be carried out before this method
        """
        pass



    

    def inference(self, inputs, Nsample=1000):
        """
        inputs has shape (Nbatch, Length), and type torch.Tensor
        return has shape (Nsample, Nbatch, Length)
        """

        Nb, _, L = inputs.size()
        output = torch.zeros((Nsample, Nb, 2))
        if torch.cuda.is_available():
            output = output.cuda()
        for n in range(Nsample):
            z = self.encode(inputs)
            output[n, :, :] = z[:,0:2]
        return output





class loss_for_ae(nn.Module):

    def __init__(self, alpha=1.0):
        super(loss_for_ae, self).__init__()
        self.alpha = alpha
        
    def forward(self, x, x_rec, pred, label):

        '''
        For now, only the case for stddiag=True is implemented.
        You will get the error if stddiag=False.
        '''
        
        #Reconstruction_loss = torch.mean((torch.abs(x_rec - x) ** 2.0) / 2.0, dim=-1)
        Reconstruction_loss = torch.mean(torch.abs(x_rec - x), dim=-1)
        parameter_loss = torch.mean(torch.abs(pred - label) / label, dim=-1)
        total_loss = Reconstruction_loss + self.alpha * parameter_loss
        return torch.mean(total_loss), torch.mean(Reconstruction_loss), torch.mean(parameter_loss)



if __name__ == '__main__':


    #filedir = '/home/tap/errorbar/testset/'
    filedir = '/home/tyamamoto/testset/'

    trainwave = np.load(filedir+'dampedsinusoid_train.npy')
    trainlabel = torch.Tensor(np.genfromtxt(filedir+'dampedsinusoid_trainLabel.dat'))

    cae = CAE_GW(8192, 32)
    cae.cuda()


    criterion = loss_for_ae(alpha=1.0)
    optimizer = optim.Adam(cae.parameters())



    for epoch in range(1):


        trainsignal = torch.Tensor(noise_inject(trainwave, pSNR=5.0))
        
        traindata = torch.utils.data.TensorDataset(trainsignal,
                                                   torch.Tensor(trainwave.reshape(-1,1,8192)),
                                                   trainlabel)
        
        data_loader = torch.utils.data.DataLoader(traindata,
                                                  batch_size=256,
                                                  shuffle=True,
                                                  num_workers=4)

        
        running_loss = 0.0
        running_rec = 0.0
        running_param = 0.0

        
        for i, data in enumerate(data_loader, 0):

            signals, waves, labels = data
            signals = signals.cuda()
            waves = waves.cuda()
            labels = labels.cuda()
            
            optimizer.zero_grad()
            outputs, preds = cae(signals)
            loss, rec, param = criterion(outputs, waves, preds, labels)
            
            running_loss += loss.data
            running_rec += rec.data
            running_param += param.data

            loss.backward()
            optimizer.step()    # Does the update

            
        print("[%2d] %.5f (= %.5f + %.5f)" % (epoch, running_loss / (i+1), running_rec / (i+1), running_param / (i+1)))


        
    '''
    with torch.no_grad():
        for _ in range(3):

            j = np.random.randint(100)
            
            inputs, labels = traindata[j]
            inputs = inputs.view(1, 1, 8192)
            inputs = inputs.cuda()
            outputs = cae(inputs)

            inputs = inputs.cpu()
            outputs = outputs.cpu()

            loss = torch.sum(torch.abs(inputs - outputs)**2) / 2.0
            print(loss)
            

            plt.figure()
            plt.plot(inputs.detach().numpy()[0,0], label='original signal')
            plt.plot(outputs.detach().numpy()[0,0], label='reproduced signal')
        
        plt.show()
    '''
