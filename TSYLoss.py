"""
TSYLoss.py
"""

from re import M
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys


class TSY_KLDiv_withStdNormal(nn.Module):
    def __init__(self):
        super(TSY_KLDiv_withStdNormal, self).__init__()

    def forward(self, mu, logvar):
        """Calculate KL divergence with standard normal distribution

        Args:
            mu (torch.tensor): mean of the approximated pdf
            logvar (torch.tensor): logarithm of the variance of the approximated pdf

        Returns:
            KL loss (torch.tensor): KL loss
        """
        return - 0.5 * torch.mean(torch.sum(1.0 + logvar - mu**2.0 - logvar.exp_(), dim=-1))




class CVAE_KLDiv(nn.Module):
    def __init__(self, eps=1e-8):
        super(CVAE_KLDiv, self).__init__()
        self.eps = eps
        
    def forward(self, mu1, logvar1, mu2, logvar2):
        var1 = logvar1.exp() + self.eps
        var2 = logvar2.exp()
        kl_div = var2/var1 + (mu1 - mu2)**2.0 / var1 + logvar1 - logvar2 - 1.0
        return torch.mean(kl_div.sum(dim=1) / 2.0)


class CVAE_LogP(nn.Module):
    def __init__(self, eps=1.0e-8):
        super(CVAE_LogP, self).__init__()
        self.eps = eps

    def forward(self, mu, logvar, label):
        var = logvar.exp()
        neg_logp = logvar + (mu - label)**2.0 / (var + self.eps) + np.log(2.0*np.pi)
        return torch.mean( neg_logp.sum(dim=1) ) / 2.0

class CVAE_LogP_withcovariance(nn.Module):
    def __init__(self, eps=1.0e-8):
        super(CVAE_LogP_withcovariance, self).__init__()
        self.eps = eps

    def forward(self, mu, logvar, r, label):
        var = logvar.exp()
        neg_logp_diag = ( logvar + (mu - label)**2.0 / (var + self.eps) + np.log(2.0*np.pi) ).sum(dim=1)
        neg_logp_offdiag = torch.log(1 - r**2.0) - 2.0 * r * (torch.sqrt(var) * (mu - label)).prod(dim=1)
        return torch.mean( neg_logp_diag + neg_logp_offdiag ) / 2.0


class CVAE_LogP_TruncatedGaussian(nn.Module):
    """
    logarithm of truncated Gaussian distribution
    """
    def __init__(self, a, eps=1.0e-5):
        super(CVAE_LogP_TruncatedGaussian, self).__init__()
        self.a = a
        self.eps = eps

    def forward(self, mu, logvar, label):

        var = logvar.exp()
        neg_logp_Gaussian = logvar + (mu - label)**2.0 / (self.eps + var) + np.log(2.0*np.pi)
        neg_logp_erfterm = torch.log( 0.5 - 0.5 * torch.erf((self.a - mu) / torch.sqrt(2.0 * var + self.eps)) )
        mask = torch.zeros_like(neg_logp_erfterm)
        mask[:,1] = 1.0
        
        return torch.mean( (neg_logp_erfterm * mask + neg_logp_Gaussian).sum(dim=1) ) / 2.0


class CVAE_logp_BCE(nn.Module):
    """
    Binary Cross Entropy loss
    """
    def __init__(self):
        super(CVAE_logp_BCE, self).__init__()
        
    def forward(self, preds, targets):
        return - torch.mean(torch.sum( targets * torch.log(preds/targets) + (1.0 - targets) * torch.log((1.0 - preds)/(1.0 - targets)), dim=1))