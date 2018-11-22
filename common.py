import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import multivariate_normal
#import load_gwdata as load


def mkdir(dir):
    if os.path.isdir(dir):
        pass
    else:
        os.makedirs(dir)
        


def noise_inject(array, pSNR, shift_max=None):
    N, L = array.shape
    if shift_max is None:
        waveformset = array
    else:
        waveformset = shift(array, shift_max)
    dataset = np.empty(waveformset.shape)
    for i in range(N):
        waveform = waveformset[i]
        peak = abs(waveform).max()
        
        if type(pSNR) is list:
            pSNR_sampled = np.random.uniform(pSNR[0], pSNR[1])
            amp = pSNR_sampled / peak
        else:
            amp = pSNR / peak

        noise = np.random.normal(0.0, 1.0, size=L)
        inject_signal = amp*waveform
        data = inject_signal + noise

        mu = data.mean()
        std = np.sqrt(data.var())
        data = (data - mu)/std
        dataset[i,:] = data

    return dataset.reshape(N, 1, L)



def shift(array, shift_max):
    N, L = array.shape
    shifted_array = np.empty((N,L))
    for n in range(N):
        n_shift = np.random.randint(-shift_max, shift_max)
        shifted_array[n] = np.roll(array[n], n_shift)
        if n_shift < 0:
            shifted_array[n, n_shift :] = 0
        else:
            shifted_array[n, : n_shift] = 0
        
        
    return shifted_array



def _noise_inject(waveform, pSNR):

    L = waveform.shape
    kmax = abs(waveform).argmax()
    peak = abs(waveform).max()
    
    if type(pSNR) is list:
        pSNR_sampled = np.random.uniform(pSNR[0], pSNR[1])
        amp = pSNR_sampled / peak
    else:
        amp = pSNR / peak
        
    noise = np.random.normal(0.0, 1.0, size=L)
    inject_signal = amp*waveform
    data = inject_signal + noise

    return data, kmax


def _pickup_ringdown(waveform, kmax, koffset, length=512):

    data = waveform[kmax-koffset : kmax-koffset+length]
    return data



def _normalize(data):
    
    mu = data.mean()
    std = np.sqrt(data.var())
    data_norm = (data - mu)/std
    return data_norm
    



def noise_inject_ringdown(waveformset, pSNR, length=512, bandpass=False):

    N, L = waveformset.shape
    dataset = np.empty((N, length))
    for i in range(N):
        #koffset = np.random.randint(0,64)
        koffset = 128
        waveform = waveformset[i]
        waveform, kmax = _noise_inject(waveform, pSNR)
        waveform = _pickup_ringdown(waveform, kmax, koffset, length=length)

        if bandpass:
            waveform = load.bandpass(waveform, 50.0, 2047.0)

        waveform = _normalize(waveform)
        dataset[i, :] = waveform

    return dataset.reshape(N,1,length)




#------------------------------------------------------------
# the function returning contour threshold 
#------------------------------------------------------------

def sigma(cov):

    eig1, eig2 = eig(cov)
    norm = 1.0 / (2*np.pi*np.sqrt(eig1*eig2))
    s3 = 0.01 / norm
    s5 = 1. / 1744278. / norm

    return [s3, s5]






if __name__=='__main__':

    Xin, Yin = np.mgrid[0:201, 0:201]
    data = gaussian2d(3.0, 100, 100, 40, 50, 0.0)(Xin, Yin) + np.random.random(Xin.shape)

    plt.matshow(data, cmap=plt.cm.gist_earth_r)

    params = fitgaussian(data)
    fit = gaussian2d(*params)

    plt.contour(fit(*np.indices(data.shape)), cmap=plt.cm.copper)
    ax = plt.gca()
    (height, x, y, cov_xx, cov_yy, cov_xy) = params

    plt.text(0.95, 0.05, """
    x : %.1f
    y : %.1f
    cov_xx : %.1f
    cov_yy : %.1f
    cov_xy: %.1f""" %(x, y, np.sqrt(cov_xx), np.sqrt(cov_yy), np.sqrt(abs(cov_xy))),
            fontsize=16, horizontalalignment='right',
            verticalalignment='bottom', transform=ax.transAxes)
    
    plt.savefig('figure.png')
    
