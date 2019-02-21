import numpy as np
import matplotlib.pyplot as plt
import os



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
        for c in range(C):
            waveform = waveformset[n,c]
            data, _ = _noise_inject(waveform, pSNR, mode)
            data = _normalize(data)
            dataset[n,c,:] = data

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
    "waveform" should have a shape (L,) 
    '''
    assert mode is 'stdfix' or 'ampfix', 'invalid mode: use "stdfix" or "ampfix".'

    L = waveform.shape
    kmax = abs(waveform).argmax()
    peak = abs(waveform).max()


    if mode=='stdfix':
    
        if type(pSNR) is list:
            pSNR_sampled = np.random.uniform(pSNR[0], pSNR[1])
            amp = pSNR_sampled / peak
        else:
            amp = pSNR / peak
        
        noise = np.random.normal(0.0, 1.0, size=L)
        inject_signal = amp*waveform
        data = inject_signal + noise


    elif mode=='ampfix':
        if type(pSNR) is list:
            pSNR_sampled = np.random.uniform(pSNR[0], pSNR[1])
            sigma = (peak / pSNR_sampled )
        else:
            sigma = (peak / pSNR)
        
        noise = np.random.normal(0.0, sigma, size=L)
        data = waveform + noise
        
    return data, kmax




def _normalize(data):
    
    mu = data.mean()
    std = np.sqrt(data.var())
    data_norm = (data - mu)/std
    return data_norm
    


data1 = np.load(os.environ['HOME']+'/gwdata/TestEOB_hPlus.npy')
data2 = np.load(os.environ['HOME']+'/gwdata/TestEOB_hPlus.npy')
print(data1.shape, data2.shape)

data_injected, _ = noise_inject([data1, data2], pSNR=30.0, shift_max=256)

print(data_injected.shape)

plt.figure()
plt.plot(data_injected[100,0])
plt.plot(data_injected[100,1])
plt.show()
