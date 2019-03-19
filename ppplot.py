import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def threshold(alpha):

    beta2 = -2.0 * np.log(1.0-alpha)
    return beta2

def sample2beta(sample, mu, Sigma):

    Lambda = np.linalg.inv(Sigma)
    beta2s = np.einsum('ij,jk,ik->i', sample-mu, Lambda, sample-mu)
    return beta2s


mu = [1.0, 1.0] 
Sigma = [[3.0, 0.2],[0.2, 2.0]]
norm = stats.multivariate_normal(mean=mu, cov=Sigma)
sample = norm.rvs(size=10000)
betas = sample2beta(sample, mu, Sigma)


plist = []
alist = np.arange(0.0, 1.0, 0.01)
for alpha in alist:

    beta_th = threshold(alpha)
    plist.append(np.sum(betas<beta_th)/10000)

plt.figure()
plt.plot(alist, alist, 'k')
plt.plot(alist, plist, 'o', c='r', markersize=3)
plt.show()
