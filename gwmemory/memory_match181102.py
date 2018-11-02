from pycbc.waveform import get_td_waveform
from pycbc.psd.analytical import KAGRA
from memory_waveform import memory_waveform
from pycbc.filter.matchedfilter import sigmasq

import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.pardir)
from PSDs import DECIGO



def inner_product(f, h, psd, df, frange=None):

    '''
    h should have only positive frequency.
    '''

    if frange is None:
        pass
    else:
        kmin = np.argmin(f[f>frange[0]])
        kmax = np.argmax(f[f<frange[1]])
        f = f[kmin:kmax]
        h = h[kmin:kmax]
        psd = psd[kmin:kmax]

    psd[psd==0] = 1e+40
    
    norm = 4*df
    integrand = h * h.conjugate() / psd
    sigmasq = norm * simps(integrand, f)
    return np.real(sigmasq)
    





if __name__ == '__main__':


    fs = 4096
    fmin = 20.0
    m1 = 36.0
    m2 = 29.0
    a1 = 0.0
    a2 = 0.0
    dist = 410.0
    #inc = 0.0
    inc = 7.0 * np.pi / 9.0    # For GW150914, the inclination angle is about 2.856rad.

    Mpc = 3.085677581 * 1e+24    #cgs
    c = 2.99792458 * 1e+10    #cgs
    
    
    hp, hc = get_td_waveform(approximant = 'TaylorT2',
                             mass1 = m1,
                             mass2 = m2,
                             spin1z = a1,
                             spin2z = a2,
                             inclination = inc,
                             distance = dist,
                             delta_t = 1.0 / fs,
                             f_lower = fmin)


    duration = hp.duration
    print('duration: ', duration, '[sec]')
    h = np.array(hp) + 1.0j * np.array(hc)
    
    
    t, hp_memory, h_mem = memory_waveform(approximant = 'TaylorT2',
                                          mass1 = m1,
                                          mass2 = m2,
                                          spin1z = a1,
                                          spin2z = a2,
                                          distance = dist,
                                          inclination = inc,
                                          delta_t = 1.0 / fs,
                                          f_lower = fmin,
                                          return_memory=True,
                                          return_as_numpy=True)



    plt.figure()
    plt.plot(t, h_mem, lw=2)
    plt.xlabel('time [sec]')
    plt.ylabel('strain')
    plt.xlim([-0.2, 0.05])
    plt.grid()

    # Generate the aLIGO ZDHP PSD

    delta_f = 1.0 / duration
    flen = len(hp)/2 + 1

    
    f = np.fft.fftfreq(len(h), 1.0/fs)
    delta_f = f[1] - f[0]
    hf = np.fft.fft(h) * delta_f

    hf = hf[f>0.0]
    f = f[f>0.0]

    psd_decigo = DECIGO(f)

    psd_kagra = KAGRA(flen, delta_f, f[1])
    psd_kagra = np.interp(f, psd_kagra.sample_frequencies, psd_kagra)
    
    print('SNR:', np.sqrt(inner_product(f, hf, psd_decigo, delta_f, [50.0, 2048.0])))

    

    normstrain = np.abs(hf) * (f**0.5)
    
    plt.figure()
    plt.loglog(f, normstrain, lw=2)
    plt.loglog(f, np.sqrt(abs(psd_decigo)), 'k', lw=2, label='DECIGO')
    plt.loglog(f, np.sqrt(abs(psd_kagra)), 'gray', lw=2, label='KAGRA')
    plt.xlim([1e-3, 1e+2])
    plt.legend()
    plt.grid()
    
    plt.show()

    
    
    '''
    hp_memory_f = hp_memory.to_frequencyseries()
    h_mem_f = h_mem.to_frequencyseries()
    rho2 = sigmasq(h_mem, psd, 20.0, 1024.0)
    print(rho2)
    
    
    plt.figure()
    plt.loglog(hp_memory_f.sample_frequencies, abs(hp_memory_f*(hp_memory_f.sample_frequencies**0.5)), 'gray', lw=1)
    plt.loglog(h_mem_f.sample_frequencies, abs(h_mem_f*(h_mem_f.sample_frequencies**0.5)), lw=2)
    plt.loglog(psd.sample_frequencies, np.sqrt(abs(psd)), 'k', lw=2)
    plt.xlim([10.0, 2049.0])
    plt.show()
    '''
