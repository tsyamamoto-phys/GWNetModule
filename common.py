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



def sigma(cov):

    w, _ = np.linalg.eig(cov)
    norm = 1.0 / (2.0*np.pi*np.sqrt(w[0]*w[1]))
    print(w)
    s3 = 0.01 * norm
    s5 = 1.0 / 1744278 * norm

    return [s3, s5]




if __name__=='__main__':


    cov = np.array([[1.0, 0.3], [0.3, 0.5]])
    cont = sigma(cov)
    
    x, y = np.mgrid[-1:1:.01, -1:1:.01]
    pos = np.empty(x.shape + (2,))
    pos[:,:,0] = x
    pos[:,:,1] = y
    rv = multivariate_normal([0.5, -0.2], [[1.0, 0.3], [0.3, 0.5]])

    plt.figure()
    plt.contour(x, y, rv.pdf(pos))
    plt.colorbar()
    plt.show()
