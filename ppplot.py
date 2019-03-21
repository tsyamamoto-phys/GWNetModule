import numpy as np
from scipy import stats
import argparse


parser = argparse.ArgumentParser(description="parse args")
parser.add_argument('--resultdir', type=str)
args = parser.parse_args()


def threshold(alpha):

    beta2 = -2.0 * np.log(1.0-alpha)
    return beta2

def sample2beta(sample, mu, Sigma):

    Lambda = np.linalg.inv(Sigma)
    beta2s = np.einsum('aj,ajk,ak->a', sample-mu, Lambda, sample-mu)
    return beta2s



datadir = args.resultdir
data = np.genfromtxt(datadir+'test_outputs_4.0.txt')
N = data.shape[0]

sample = np.empty((N,2))
mu = np.empty((N, 2))
Sigma = np.empty((N,2,2))

for n in range(N):
    mu[n] = data[n,0:2]
    sample[n] = data[n,2:4]
    Sigma[n,0,0] = data[n,4]
    Sigma[n,1,1] = data[n,5]
    Sigma[n,0,1] = data[n,6]
    Sigma[n,1,0] = data[n,6]

betas = sample2beta(sample, mu, Sigma)


plist = []
#alist = np.arange(0.0, 1.0, 0.01)
alist = [0.1, 0.5, 0.9, 0.99]
for alpha in alist:

    beta_th = threshold(alpha)
    plist.append(np.sum(betas<beta_th)/N)


print(plist)

