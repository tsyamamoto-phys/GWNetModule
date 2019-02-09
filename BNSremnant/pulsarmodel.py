import numpy as np
import scipy.signal as signal
from scipy.constants import c, G
c *= 1e+2
G *= 1e+3

import matplotlib.pyplot as plt
from antenna_pattern import antenna_a, antenna_b

import pycbc.noise
import pycbc.psd


import sys
sys.path.append('GWNetModule')
from common import noise_inject


def magnetar_model_freq(duration, dt, tau, n, f0):
    t = np.arange(0.0, duration, dt)
    f = f0 * (t/tau + 1.0) ** (1.0/(1.0-n))
    return t, f



def freq2phase(t, f, phi0=0.0):
    dt = t[1] - t[0]
    phi = 2.0 * np.pi * np.cumsum(f) * dt
    return phi



def magnetar_model_amp(epsilon, Izz, d):
    dist = d * 3.085677581*1e+24
    factor = 4.0 * (np.pi**2.0) * G / (c**4.0)
    return factor * Izz * epsilon / dist



def magnetar_model(*args, **kwargs):
    duration = kwargs['duration']
    dt = kwargs['dt']
    T0 = kwargs['start_time']
    tau = kwargs['tau']
    n = kwargs['breaking_index']
    f0 = kwargs['f0']
    epsilon = kwargs['epsilon']
    Izz = kwargs['Izz']
    d = kwargs['d'] # in Mpc
    inc = kwargs['inclination']
    alpha = kwargs['alpha']
    delta = kwargs['delta']
    pol = kwargs['polarization']
    
    t, freq = magnetar_model_freq(duration, dt, tau, n, f0)
    phi = freq2phase(t, freq)
    amp = magnetar_model_amp(epsilon, Izz, d)

    h0 = amp * (freq**2.0)
    A_plus = h0 * (1. + np.cos(inc)**2.0) / 2.
    A_cross = h0 * (np.cos(2.*inc))
    
    h_plus = A_plus * np.cos(phi)
    h_cross = A_cross * np.sin(phi)
    
    F_plus = antenna_a(t, alpha, delta)*np.cos(2.*pol) \
        + antenna_b(t, alpha, delta)*np.sin(2.*pol)
    F_cross = antenna_b(t, alpha, delta)*np.cos(2.*pol) \
        - antenna_a(t, alpha, delta)*np.sin(2.*pol)

    
    h = F_plus * h_plus + F_cross * h_cross
    
    return t, h, freq



def contract(array, width=4):
    '''
    array.shape = (ny, nx)
    '''
    ny, nx = array.shape
    ny_c = int(ny/width)
    array_c = np.zeros((ny_c, nx))

    for y in range(ny_c):
        array_c[y, :] = np.sum(array[y*width:(y+1)*width, :], axis=0)

    return array_c

    

def contrast01(array):
    ny, nx = array.shape
    array_c = np.zeros_like(array)
    for y in range(ny):
        if y==0:
            pass
        elif y==ny-1:
            pass
        else:
            array_c[y,:] = 2.*array[y,:] - (array[y-1,:] + array[y+1,:])

    return array_c


def image_normalize(img):
    return img / img.max()
    




if __name__=='__main__':


    t, f = magnetar_model_freq(duration=10.0,
                               dt=1.0/4096,
                               tau=30.0*1e-3,
                               n=5,
                               f0=1000.0)

    phi = freq2phase(t, f)


    t, h, f = magnetar_model(duration=500.0,
                             dt=1.0/4096,
                             start_time=0.0,
                             tau=1.0e+2,
                             breaking_index=2.5,
                             f0=1000.0,
                             epsilon=1.0e-3,
                             Izz=4.34e+45,
                             d=1.0,
                             inclination=0.0,
                             alpha=0.0,
                             delta=0.0,
                             polarization=0.0)


    flow = 10.0
    delta_f = 1.0 / 16
    flen = int(2048 / delta_f) + 1
    psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)

    delta_t = 1.0 / 4096
    tsamples = int(500 / delta_t)
    ts = pycbc.noise.noise_from_psd(tsamples, delta_t, psd, seed=127)
    s = h + np.array(ts)
    
    plt.figure()
    plt.plot(t, s)
    plt.plot(t, h)

    
    
    #h_inj, _ = noise_inject(h.reshape(1, -1), pSNR=1.0, mode='ampfix')
    
    #freq, time, spec = signal.spectrogram(s, fs=4096, nperseg=4096)
    freq, time, spec = signal.stft(h, fs=4096, nperseg=4096)    

    spec = spec[freq>250.0]
    freq = freq[freq>250.0]
    print(spec.shape)

    quasi_phase = abs(np.real(spec)-np.imag(spec))
    
    plt.figure()
    plt.pcolormesh(time, freq, image_normalize(quasi_phase), cmap='gray')
    #plt.plot(t[::4096*10], f[::4096*10], 'o', markersize=1.0)
    plt.colorbar()
    plt.xlabel('time[s]')
    plt.ylabel('frequency[Hz]')

    '''
    plt.figure()
    plt.pcolormesh(time, freq, image_normalize(abs(contrast01(spec))), cmap='gray')
    plt.colorbar()
    plt.xlabel('time[s]')
    plt.ylabel('frequency[Hz]')

    
    width = 16
    plt.figure()
    plt.pcolormesh(time, freq[::width], contract(spec, width), cmap='inferno')
    #plt.plot(t[::4096*10], f[::4096*10], 'o', markersize=1.0)
    plt.xlabel('time[s]')
    plt.ylabel('frequency[Hz]')
    '''

    plt.show()
