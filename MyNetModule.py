import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DanielSimplerNet(nn.Module):
    '''
    reference: arXiv 1701.00008
    ( Daniel George and Eliu Huerta)
    This is the simpler net.
    '''

    def __init__(self, out_features):
        super(DanielSimplerNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=16)
        self.pool1 = nn.MaxPool1d(4, stride=4)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8, dilation=4)
        self.pool2 = nn.MaxPool1d(4, stride=4)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8, dilation=4)
        self.pool3 = nn.MaxPool1d(4, stride=4)

        self.dense1 = nn.Linear(in_features=64*119, out_features=64)
        self.dense2 = nn.Linear(in_features=64, out_features=out_features)


    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = x.view(-1, 64*119)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x
    

    
    
    
class DanielDeeperNet(nn.Module):
    '''
    reference: arXiv 1701.00008
    ( Daniel George and Eliu Huerta)
    This is the deeper net.
    '''

    def __init__(self, out_features):
        super(DanielDeeperNet, self).__init__()
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
        self.dense3 = nn.Linear(in_features=64, out_features=out_features)


    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = F.relu(self.pool4(self.conv4(x)))
        x = x.view(-1, 512*14)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x




def _cal_length(N, k, s=1, d=1):
    return math.floor((N - d*(k-1) -1) / s + 1)


class RingdownNet(nn.Module):

    def __init__(self, length=512, out_features=2):
        super(RingdownNet, self).__init__()
        self.length = length

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=4)
        self.L = _cal_length(self.length, 4)
        #self.pool1 = nn.MaxPool1d(4, stride=4)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4)
        self.L = _cal_length(self.L, 4)

        self.pool2 = nn.MaxPool1d(4, stride=4)
        self.L = _cal_length(self.L, 4, s=4)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8)
        self.L = _cal_length(self.L, 8)

        #self.pool3 = nn.MaxPool1d(4, stride=4)

        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=8)
        self.L = _cal_length(self.L, 8)

        self.pool4 = nn.MaxPool1d(4, stride=4)
        self.L = _cal_length(self.L, 4, s=4)
        
        self.dense1 = nn.Linear(in_features=256*self.L, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=out_features)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.pool4(self.conv4(x)))
        x = x.view(-1, 256*self.L)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x



class ErrorbarEstimateNet(nn.Module):

    def __init__(self, out_features):
        super(ErrorbarEstimateNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=16)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=16)
        self.pool2 = nn.MaxPool1d(4, stride=4)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=16, dilation=2)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=16, dilation=2)
        self.pool4 = nn.MaxPool1d(4, stride=4)
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=32, dilation=2)

        self.dense1 = nn.Linear(in_features=1024*433, out_features=512)
        self.dense2 = nn.Linear(in_features=512, out_features=128)
        self.dense3 = nn.Linear(in_features=128, out_features=out_features)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.pool4(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = x.view(-1, 1024*433)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.softmax(self.dense3(x), dim=1)
        return x


    

#-----------------------------------------------------------
# module for training, validation and testing networks
#-----------------------------------------------------------



def trian_model(model, criterion, optimizer, data_loader, modeldir,
                epoch=0, modelsaveepoch=None):

    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
            
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(labels, *outputs)
        running_loss += loss.data
    
        loss.backward()
        optimizer.step()    # Does the update

        
    if modelsaveepoch is not None:
        if epoch % modelsaveepoch == (modelsaveepoch - 1):
            torch.save(model.state_dict(), modeldir+'model_%d.pt'%(epoch+1))
            torch.save(optimizer.state_dict(), modeldir+'optimizer_%d.pt'%(epoch+1))

    return model, running_loss / (i+1)



def validate_model(model, criterion, inputs, labels):

    with torch.no_grad():
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = model(inputs)
        loss = criterion(labels, *outputs)
        
    return loss








class MeanRelativeError(nn.Module):
    '''
    In PyTorch, mean relative error is not defined as loss function.
    Here, I implemented the relative error.
    '''

    def __init__(self):
        super(MeanRelativeError, self).__init__()
        
    def forward(self, outputs, labels):
        relerr_for_eachdata = torch.sum(torch.abs(outputs - labels) / labels, dim=1)
        mre = torch.mean(relerr_for_eachdata)

        return mre
    
    
    
    
    
class MassSpinConsistencyPenalty(nn.Module):
    '''
    The network has 3 outputs (m1, m2, a).
    For non-spinning binary, the analytic formula deriving the spin from mass ratio is proposed.
    (reference: arXiv xxxx.yyyyy Gair et al)
    The penalty term made from the formula is added to the l1 loss.
    '''
    
    
    def __init__(self, beta=0.01):
        super(MassSpinConsistencyPenalty, self).__init__()
        self.beta = beta
    
    def _mass2spin(self, m1, m2):
        nu = m1*m2 / (m1 + m2)**2
        a = (math.sqrt(12) * nu - 3.87 * nu**2)
        return a
        
        
    def _penalty(self, m1, m2, a):
        a_frommass = self._mass2spin(m1, m2)
        diff = torch.abs(a - a_frommass)
        return diff
    
    
    def forward(self, outputs, labels):
        m1 = outputs[:,0]
        m2 = outputs[:,1]
        a = outputs[:,2]
                
        penalty = self._penalty(m1, m2, a)
        loss = torch.sum(torch.abs(outputs - labels), dim=1) + self.beta * penalty
        
        return torch.mean(loss)
    



    
def array2vecs(array, vmin, vmax, bins):
    array = array.astype(np.float32)
    N = array.shape[0]
    vecs = np.zeros((N, bins), dtype=int)
    dv = (vmax - vmin) / bins
    bin_array = np.arange(vmin, vmax+dv/2., dv)
    for j in range(N):
        value = array[j]
        for i in range(bins):
            binmin = bin_array[i]
            binmax = bin_array[i+1]
            if (binmin<=value) and (value<binmax):
                vecs[j, i] = 1
                break


    assert vecs.sum(axis=1).all()==np.ones(N).all(), "Incorrectly labeled"
    return vecs

    


    
if __name__ == '__main__':
    
    print(' This is the demonstration.')
    '''
    This main function is for checking the module.
    '''


    def func(x):
        return x, x**2

    def func2(x, y):
        return x+y

    z = func(2)
    print(func2(*z))

