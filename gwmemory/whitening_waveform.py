from pycbc.waveform import get_td_waveform
from pycbc.psd.analytical import KAGRA
from memory_waveform import memory_waveform
from pycbc.filter.matchedfilter import sigmasq

import numpy as np
from scipy.integrate import simps
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


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



hp, hc = get_td_waveform(approximant = 'SEOBNRv4',
                         mass1 = m1,
                         mass2 = m2,
                         spin1z = a1,
                         spin2z = a2,
                         distance = dist,
                         inclination = inc,
                         delta_t = 1.0 / fs,
                         f_lower = fmin)
                             



t, hm, hmem = memory_waveform(approximant = 'SEOBNRv4',
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
plt.plot(hp.sample_times, hp)
plt.plot(t, hm)
plt.grid()
plt.xlabel('time[sec]')
plt.ylabel('strain')
plt.title('unwhitened')




delta_f = 1.0 / 10.0    # This psd should be interpolated in following.
flen = int(4096 / delta_f)
low_frequency_cutoff = 0.0
psd = KAGRA(flen, delta_f, low_frequency_cutoff)
psd[0] = 1e+10
interp_psd = interp1d(psd.sample_frequencies, psd)


freq = np.fft.rfftfreq(len(hp), 1.0/fs)

# function to whiten data
def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    freqs1 = np.linspace(0,2048.,Nt/2+1)

    # whitening: transform to freq domain, divide by asd, then transform back, 
    # taking care to get normalization right.
    hf = np.fft.rfft(strain)
    norm = 1./np.sqrt(1./(dt*2))
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht


# now whiten the data from H1 and L1, and the template (use H1 PSD):
hm_whiten = whiten(hm, interp_psd, dt=1.0/fs)
hp_whiten = whiten(hp, interp_psd, dt=1.0/fs)

plt.figure()
plt.plot(t, hm_whiten)
plt.plot(t, hp_whiten)
plt.xlabel('time [sec]')
plt.ylabel('strain')
plt.grid()
plt.title('whitened')

plt.show()



'''
# We need to suppress the high frequency noise (no signal!) with some bandpassing:
bb, ab = butter(4, [fband[0]*2./fs, fband[1]*2./fs], btype='band')
normalization = np.sqrt((fband[1]-fband[0])/(fs/2))
strain_H1_whitenbp = filtfilt(bb, ab, strain_H1_whiten) / normalization
strain_L1_whitenbp = filtfilt(bb, ab, strain_L1_whiten) / normalization
'''
