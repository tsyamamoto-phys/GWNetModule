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

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=8)
        self.L = _cal_length(self.length, 8)
        #self.pool1 = nn.MaxPool1d(4, stride=4)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8)
        self.L = _cal_length(self.L, 8)

        #self.pool2 = nn.MaxPool1d(4, stride=4)
        #self.L = _cal_length(self.L, 4, s=4)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8)
        self.L = _cal_length(self.L, 8)

        #self.pool3 = nn.MaxPool1d(4, stride=4)

        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=16)
        self.L = _cal_length(self.L, 16)

        #self.pool4 = nn.MaxPool1d(4, stride=4)
        #self.L = _cal_length(self.L, 4, s=4)
        
        self.conv5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=16)
        self.L = _cal_length(self.L, 16)


        self.dense1 = nn.Linear(in_features=256*self.L, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=out_features)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #x = F.relu(self.pool4(self.conv4(x)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(-1, 256*self.L)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x



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
        self.dense3 = nn.Linear(in_features=64, out_features=5)



    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = F.relu(self.pool4(self.conv4(x)))
        x = x.view(-1, 512*14)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)

        preds = x[:, 0:2]
        y = x[:, 2:5]
        Lambda = torch.zeros_like(y)
        Lambda[:,0] = y[:,0]**2.0 + y[:,1]**2.0
        Lambda[:,1] = y[:,1]*(y[:,0] + y[:,2])
        Lambda[:,2] = y[:,1]**2.0 + y[:,2]**2.0
        return preds, Lambda


    

#-----------------------------------------------------------
# module for training, validation and testing networks
#-----------------------------------------------------------



def train_model(model, criterion, optimizer, data_loader, modeldir,
                epoch=0, modelsaveepoch=None):

    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
            
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(labels, outputs)
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
        loss, _kl, _rec = criterion(labels, *outputs)
        
    return loss








class MeanRelativeError(nn.Module):
    '''
    In PyTorch, mean relative error is not defined as loss function.
    Here, I implemented the relative error.
    '''

    def __init__(self):
        super(MeanRelativeError, self).__init__()
        
    def forward(self, outputs, labels):
        relerr_for_eachdata = torch.sum(torch.abs((outputs - labels) / labels), dim=1)
        mre = torch.mean(relerr_for_eachdata)

        return mre
    

class errorbarError(nn.Module):

    def __init__(self, a=1.0, d=1.0):
        super(errorbarError, self).__init__()
        self.a = a
        self.d = d

    def forward(self, df_t, df_p):
        
        mse = torch.sum(torch.tanh((df_t-df_p)/self.d), dim=-1)
        reg = torch.sum((df_t-df_p)**2.0, dim=-1)

        return torch.mean(mse + self.a*reg)
 




class log_gaussian_error(nn.Module):

    def __init__(self, alpha=1.0):
        super(log_gaussian_error, self).__init__()
        self.alpha = alpha


    def _normalize_square_sum(self, df, Lambda):
        
        f0f0 = df[:,0] * df[:,0]
        f0f1 = df[:,0] * df[:,1]
        f1f1 = df[:,1] * df[:,1]
        
        # diagonal case
        #a = f0f0*Lambda[:,0] + f1f1*Lambda[:,1]

        # non diagonal case
        a = f0f0 * Lambda[:,0] + 2 * f0f1 * Lambda[:,1] + f1f1 * Lambda[:,2]
        return a
    

    def forward(self, preds, Lambda, labels):
        
        det = torch.abs(Lambda[:,1]*Lambda[:,1] - Lambda[:,0]*Lambda[:,2])
        predloss = self._normalize_square_sum(preds-labels, Lambda)
        return torch.mean(-self.alpha*torch.log(det)/2. + predloss) 



class cdf_error(nn.Module):

    # based on chi-square distribution w\ d.o.f = 2
    
    def __init__(self, alpha=0.0, prederr='MSE'):
        super(cdf_error, self).__init__()
        self.alpha = alpha
        self.chi2_cdf = (lambda x: 1-torch.exp(-x/2.0))
        self.prederr = prederr

    def _sigma2lambda(self, sigma):
        """
        sigma is the half of Sigma(covariance matrix),
        and should be given in size(Nb, l(l+1)/2).
        For l=2,
        sigma[:,0] = sigma_11,
        sigma[:,1] = sigma_12,
        sigma[:,2] = sigma_22.
        """

        Sigma = torch.zeros_like(sigma)
        Lambda = torch.zeros_like(sigma)

        Sigma[:,0] = sigma[:,0]**2.0 + sigma[:,1]**2.0
        Sigma[:,1] = sigma[:,1]*(sigma[:,0] + sigma[:,2])
        Sigma[:,2] = sigma[:,1]**2.0 + sigma[:,2]**2.0
        det = Sigma[:,1]*Sigma[:,1] - Sigma[:,0]*Sigma[:,2]
        
        Lambda[:,0] = Sigma[:,2] / det
        Lambda[:,1] = -Sigma[:,1] / det
        Lambda[:,2] = Sigma[:,0] / det
        return Lambda


    def _normalize_square_sum(self, df, Lambda):
        
        f0f0 = df[:,0] * df[:,0]
        f0f1 = df[:,0] * df[:,1]
        f1f1 = df[:,1] * df[:,1]
        
        # diagonal case
        #a = f0f0*Lambda[:,0] + f1f1*Lambda[:,1]

        # non diagonal case
        a = f0f0 * Lambda[:,0] + 2 * f0f1 * Lambda[:,1] + f1f1 * Lambda[:,2]
        return a
    


    def _cdf_error(self, df, Lambda):
        #Lambda = self._sigma2lambda(sigma)
        y = self._normalize_square_sum(df, Lambda)
        y_sort, _ = torch.sort(y)
        cdf_t = self.chi2_cdf(y_sort)

        Nb = df.size()[0]
        cdf_e = torch.arange(0.0, 1.0, 1.0/Nb, out=torch.FloatTensor())
        if torch.cuda.is_available(): cdf_e = cdf_e.cuda()
        error = torch.sum((cdf_t - cdf_e)**2.0) / 2.0
        return error

    
    def forward(self, preds, Lambda, labels):
        if self.prederr=='MSE':
            prediction_error = torch.mean(torch.sum((preds - labels)**2.0, dim=-1) / 2.0)

        elif self.prederr=='MRE':
            prediction_error = torch.mean(torch.sum(torch.abs(preds - labels) / labels, dim=-1))
       
        elif self.prederr=='L1':
            prediction_error = torch.mean(torch.sum(torch.abs(preds - labels), dim=-1))

        elif self.prederr=='maharanobis':
            prediction_error = torch.mean(self._normalize_square_sum(preds-labels, Lambda))
       
        else:
            pass

        cdf_error = self._cdf_error(preds - labels, Lambda)

        return prediction_error + self.alpha * cdf_error, prediction_error, cdf_error 
        
    
    
    
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

