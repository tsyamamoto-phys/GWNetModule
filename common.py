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
# Gaussian Fitting (2d)
#------------------------------------------------------------

def gaussian2d(height, x_mean, y_mean, cov_xx, cov_yy, cov_xy):

    detcov = cov_xx * cov_yy - cov_xy**2.0
    gam_xx = cov_yy / detcov
    gam_yy = cov_xx / detcov
    gam_xy = -cov_xy / detcov
    #norm = 1.0 / (2.0*np.pi*(abs(detcov)**0.5))

    return lambda x,y: height * np.exp( - (x-x_mean)**2.0*gam_xx/2.0 - (y-y_mean)**2.0*gam_yy/2.0 - 2.0*(x-x_mean)*(y-y_mean)*gam_xy)
    
'''
def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments 
    data has the shape (N, 2). N is the number of the data.
    """
    
    total = data.sum()
    data_norm = data / total
    N = data.shape[0]
    
    mean = np.mean(data_norm, axis=0)
    cov = np.cov(data_norm.T)
    
    height = data.max()
    
    print(height, mean[0], mean[1], cov[0,0], cov[1,1], cov[0,1])
    return height, mean[0], mean[1], cov[0,0], cov[1,1], cov[0,1]
'''



def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    width = 0.0
    return height, x, y, width_x, width_y, width


def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian2d(*p)(*np.indices(data.shape)) - data)
    p, success = optimize.leastsq(errorfunction, params)
    return p



if __name__=='__main__':


    x, y = np.mgrid[-1:1:.01, -1:1:.01]
    pos = np.empty(x.shape + (2,))
    pos[:,:,0] = x
    pos[:,:,1] = y
    rv = multivariate_normal([0.5, -0.2], [[1.0, 0.3], [0.3, 0.5]])

    plt.figure()
    plt.contour(x, y, rv.pdf(pos))
    plt.colorbar()
    plt.show()
















    '''

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
    '''
