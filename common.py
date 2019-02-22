import numpy as np
import matplotlib.pyplot as plt
import os

#import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import multivariate_normal
import load_gwdata as load


def mkdir(dir):
    if os.path.isdir(dir):
        pass
    else:
        os.makedirs(dir)
        



def noise_inject(array_list, pSNR, shift_max=None, mode='stdfix'):
    '''
    mode: ampfix or stdfix
    each components of array_list are input signals from different channels
    '''

    assert mode is 'stdfix' or 'ampfix', 'invalid mode: use "stdfix" or "ampfix"'
    
    C = len(array_list)
    N, L = array_list[0].shape
    dataset = np.zeros((N, C, L))
    waveformset = np.zeros((N, C, L))

    if shift_max is None:
        for n in range(N):
            for c in range(C):
                waveformset[n, c, :] = array_list[c][n,:]

    else:
        waveformset = shift(array_list, shift_max)



    dataset = np.empty(waveformset.shape)
    for n in range(N):
        waveform = waveformset[n]
        data, _ = _noise_inject(waveform, pSNR, mode)
        data = _normalize(data)
        dataset[n] = data

    return dataset, waveformset 





def shift(array_list, shift_max):
    C = len(array_list)
    N, L = array_list[0].shape
    shifted_array = np.empty((N,C,L))
    for n in range(N):
        n_shift = np.random.randint(-shift_max, shift_max)
        
        for c in range(C):
            shifted_array[n,c] = np.roll(array_list[c][n], n_shift)
            if n_shift < 0:
                shifted_array[n, c, n_shift :] = 0
            else:
                shifted_array[n, c, : n_shift] = 0
            
        
    return shifted_array




def _noise_inject(waveform, pSNR, mode='stdfix'):

    '''
    This method will treat one waveform.
    "waveform" should have a shape (C, L)
    '''
    assert mode is 'stdfix' or 'ampfix', 'invalid mode: use "stdfix" or "ampfix".'

    C, L = waveform.shape
    kmax = abs(waveform[0]).argmax()
    peak = abs(waveform[0]).max()


    if mode=='stdfix':
    
        if type(pSNR) is list:
            pSNR_sampled = np.random.uniform(pSNR[0], pSNR[1])
            amp = pSNR_sampled / peak
        else:
            amp = pSNR / peak
        
        noise = np.random.normal(0.0, 1.0, size=(C,L))
        inject_signal = amp*waveform
        data = inject_signal + noise


    elif mode=='ampfix':
        if type(pSNR) is list:
            pSNR_sampled = np.random.uniform(pSNR[0], pSNR[1])
            sigma = (peak / pSNR_sampled )
        else:
            sigma = (peak / pSNR)
        
        noise = np.random.normal(0.0, sigma, size=(C,L))
        data = waveform + noise
        
    return data, kmax




def _normalize(data):
    
    C, L = data.shape
    data_norm = np.zeros_like(data)

    for c in range(C):
        mu = data[c].mean()
        std = np.sqrt(data[c].var())
        data_norm[c] = (data[c] - mu)/std
    return data_norm
    




def _pickup_ringdown(waveform, kmax, koffset, length=512):

    """
    waveform should have a shape (C, L).
    """
    data = waveform[:, kmax-koffset : kmax-koffset+length]
    return data



def noise_inject_ringdown(array_list, pSNR, length=512, bandpass=False, mode='stdfix'):
    
    C = len(array_list)
    N, L = array_list[0].shape
    array = np.zeros((N,C,L))

    for n in range(N):
        for c in range(C):
            array[n,c,:] = array_list[c][n]

    dataset = np.zeros((N, C, length))

    for n in range(N):
        koffset = 256 + np.random.randint(-20,20)
        waveform = array[n]
        waveform, kmax = _noise_inject(waveform, pSNR, mode)
        waveform = _pickup_ringdown(waveform, kmax, koffset, length=length)

        if bandpass:
            waveform = load.bandpass(waveform, 100.0, 1024.0)

        waveform = _normalize(waveform)
        dataset[n] = waveform

    return dataset





def many_noise_inject(waveform, pSNR, N=1000, mode='stdfix'):

    L = waveform.shape[-1]
    dataset = np.empty((N, L))
    for i in range(N):
        data, _ = _noise_inject(waveform, pSNR, mode)
        data = _normalize(data)
        dataset[i,:] = data

    return dataset.reshape(N, 1, L)




def sigma(cov):

    w, _ = np.linalg.eig(cov)
    norm = 1.0 / (2.0*np.pi*np.sqrt(w[0]*w[1]))

    p50 = 0.5 * norm
    p75 = 0.25 * norm
    p90 = 0.01 * norm
    return [float(p90), float(p75), float(p50)]


    """
    s1 = 1 / 3.1514872 * norm
    s3 = 0.01 * norm
    s5 = 1.0 / 1744278 * norm

    return [float(s5), float(s3), float(s1)]
    """





if __name__=='__main__':

    def down_sampling(array):

        '''
        array have a shape (N, L)
        '''
        array_ds = array[:,::2]
        return array_ds

    data1 = down_sampling(np.load(os.environ['HOME']+'/gwdata/TrainEOB_hPlus.npy'))
    data2 = down_sampling(np.load(os.environ['HOME']+'/gwdata/TrainEOB_hCross.npy'))

    plt.figure()
    plt.plot(data1[1])
    plt.plot(data2[1])
    plt.show()


    data_injected = noise_inject_ringdown([data1, data2], pSNR=10.0)
    print(data_injected.shape)
    
    plt.figure()
    plt.plot(data_injected[100, 0])
    plt.plot(data_injected[100, 1])
    plt.show()
    
