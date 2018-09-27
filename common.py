import numpy as np
import os


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
    



def noise_inject_ringdown(waveformset, pSNR, length=512):

    N, L = waveformset.shape
    dataset = np.empty((N, length))
    for i in range(N):
        koffset = np.random.randint(0,256)
        waveform = waveformset[i]
        waveform, kmax = _noise_inject(waveform, pSNR)
        waveform = _pickup_ringdown(waveform, kmax, koffset, length=length)
        waveform = _normalize(waveform)
        dataset[i, :] = waveform

    return dataset.reshape(N,1,length)







if __name__=='__main__':
    hp = np.load('TestEOB_hPlus.npy')
    data = noise_inject(hp, 4.0)
    print(data.shape, hp.shape)

