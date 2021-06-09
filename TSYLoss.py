import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class CVAE_KLDiv(nn.Module):
    def __init__(self):
        super(CVAE_KLDiv, self).__init__()
        
    def forward(self, mu1, logvar1, mu2, logvar2):

        var1 = logvar1.exp()
        var2 = logvar2.exp()
        
        kl_div = var2/var1 + (mu1 - mu2)**2.0 / var1 + logvar1 - logvar2 - 1.0

        return torch.mean(kl_div.sum(dim=1) / 2.0)



class CVAE_LogP(nn.Module):
    def __init__(self):
        super(CVAE_LogP, self).__init__()

    def forward(self, mu, logvar, label):

        var = logvar.exp()
        neg_logp = logvar + (mu - label)**2.0 / var + np.log(2.0*np.pi)
        
        return torch.mean( neg_logp.sum(dim=1) ) / 2.0
    

