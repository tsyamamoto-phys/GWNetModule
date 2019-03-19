import torch
import torch.nn as nn
import torch.nn.functional as F


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



    def forward(self, inputs):
        x = F.relu(self.pool1(self.conv1(inputs)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = F.relu(self.pool4(self.conv4(x)))
        x = x.view(-1, 512*14)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)

        preds = x[:,0:2]
        sigma2 = torch.abs(x[:,2:4])
        r = torch.sigmoid(x[:,4])
        return preds, sigma2, r



class ErrorEstimateNet_double(nn.Module):

    def __init__(self):
        super(ErrorEstimateNet_double, self).__init__()
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
        self.dense3 = nn.Linear(in_features=64, out_features=2)
        """
        self.dense1_sigma = nn.Linear(in_features=512*14, out_features=128)
        self.dense2_sigma = nn.Linear(in_features=128, out_features=64)
        self.dense3_sigma = nn.Linear(in_features=64, out_features=3)
        """
        self.dense1_sigma = nn.Linear(in_features=512*14, out_features=64)
        self.dense2_sigma = nn.Linear(in_features=64, out_features=3)


    def point_pred(self, inputs):
        x = F.relu(self.pool1(self.conv1(inputs)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = F.relu(self.pool4(self.conv4(x)))
        x = x.view(-1, 512*14)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)

        return x


    def variance(self, inputs):

        x = F.relu(self.pool1(self.conv1(inputs)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = F.relu(self.pool4(self.conv4(x)))
        x = x.view(-1, 512*14)
        x = F.relu(self.dense1_sigma(x))
        x = self.dense2_sigma(x)

        sigma_11 = x[:,0]**2.0 + x[:,1]**2.0
        sigma_22 = x[:,1]**2.0 + x[:,2]**2.0
        sigma_12 = x[:,1]*(x[:,0] + x[:,2])

        return sigma_11, sigma_22, sigma_12 



    def freeze_pred_net(self):

        self.conv1.weight.requires_grad = False
        self.conv1.bias.requires_grad = False
        self.conv2.weight.requires_grad = False
        self.conv2.bias.requires_grad = False
        self.conv3.weight.requires_grad = False
        self.conv3.bias.requires_grad = False
        self.conv4.weight.requires_grad = False
        self.conv4.bias.requires_grad = False
        self.dense1.weight.requires_grad = False
        self.dense1.bias.requires_grad = False
        self.dense2.weight.requires_grad = False
        self.dense2.bias.requires_grad = False
        self.dense3.weight.requires_grad = False
        self.dense3.bias.requires_grad = False


class log_likelihood_error(nn.Module):

    # based on chi-square distribution w\ d.o.f = 2
    
    def __init__(self):
        super(log_likelihood_error, self).__init__()

    def forward(self, label, pred, sigma2, r):
        """
        sigma2 should be vector
        """
        df = pred - label
        mse = (torch.sum(df.pow(2.0) / sigma2, dim=-1) - 2.0 * r * torch.prod(df, dim=-1) / torch.sqrt(torch.prod(sigma2, dim=-1))) / (1-r**2.0)
        rgl = torch.log(1.0-r**2.0) + torch.sum(torch.log(sigma2), dim=-1)
        return torch.mean(mse +rgl)


class reducedNLL(nn.Module):

    # based on chi-square distribution w\ d.o.f = 2
    
    def __init__(self):
        super(reducedNLL, self).__init__()

    def forward(self, label, pred, var):
        """
        sigma2 should be vector
        """
        df = pred - label
        err11 = torch.abs(var[0] - df[:,0]*2.0)
        err22 = torch.abs(var[1] - df[:,1]*2.0)
        err12 = torch.abs(var[2] - df[:,0]*df[:,1])
        
        return torch.mean(err11+err22+err12)


