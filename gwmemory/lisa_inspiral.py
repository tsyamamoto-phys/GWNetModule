import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import tukey, spectrogram
from scipy.constants import G, c
from scipy.special import gamma, fresnel

import pycbc.noise
import pycbc.psd

PI = np.pi
PI2 = PI*PI
Msun = 1.9884 * 1e+30
Mpc = 3e+22

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


def inspiral(t, f0, Mc, phi0=0.0, dist=1.0):

    
    df = (96* PI**(8.0/3.0) / 5) * f0**(11.0/3.0) * (G*Mc*Msun/(c**3.0))**(5.0/3.0)
    amp = 5.0 * df / (96 * PI2 * f0**3.0 * dist * Mpc) * c
    phi = 2*PI*(f0+0.5*df*t)*t + phi0
    hp = amp * np.cos(phi)
    hc = -amp * np.sin(phi)

    return hp, hc, amp, phi, df


def inspiral02(t, f0, df, amp=1.0, phi0=0.0):

    phi = 2*np.pi*(f0+0.5*df*t)*t + phi0
    hp = amp * np.cos(phi)
    hc = amp * np.sin(phi)

    return hp, hc, phi


def integralsin2(x):

    fresnelc, fresnels = fresnel(np.sqrt(2./PI)*x)
    intc = 3*x*np.sin(x**2.0)*gamma(3./4.) / (8.*gamma(7./4.)) - 3.0*np.sqrt(2.0*PI)*fresnels*gamma(3./4.) / (16.*gamma(7./4.))
    ints = -5*x*np.cos(x**2.)*gamma(5./4.)/(8.*gamma(9./4.)) + 5.*np.sqrt(2.*PI)*fresnelc*gamma(5./4.)/(16.*gamma(9./4.))

    return intc, ints


def inspiral_memory(t, f, Mc, dist=1.0, phi0=0.0):

    df = (96* PI**(8.0/3.0) / 5) * f**(11.0/3.0) * (Mc*Msun)**(5.0/3.0) * (G/c**3.0)**(5.0/3.0)
    amp = 5.0 * df / (96 * PI2 * f**3.0 * dist*Mpc) * c
    factor = dist*Mpc / c

    intc = integralsin2(np.sqrt(2*PI*df)*(t+f/df))[0] - integralsin2(np.sqrt(2*PI*df)*f/df)[0]
    ints = integralsin2(np.sqrt(2*PI*df)*(t+f/df))[1] - integralsin2(np.sqrt(2*PI*df)*f/df)[1]
    h = factor * 2*PI2*(amp**2.0) * ( (df*(t**3.0)/3.0 + f*df*(t**2.0) + f*t)\
        - np.sqrt(df/2./PI)/(2.*PI) * (np.cos(2*PI*(f**2.)/df)*intc + np.sin(2*PI*(f**2.)/df)*ints) / 2.0 )
    
    return h, df, amp





if __name__ == '__main__':

    f0 = 50.0
    fs = 4096
    t = np.arange(0.0, 500.0, 1.0/fs)
    hp, hc, amp, phi, df = inspiral(t, f0, 1.0, dist=400.)


    flow = 10.0
    delta_f = 1.0 / 16
    flen = int(2048 / delta_f) + 1
    psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)

    delta_t = 1.0 / 4096
    tsamples = int(500 / delta_t)
    ts = pycbc.noise.noise_from_psd(tsamples, delta_t, psd, seed=127)
    s = hp + np.array(ts)

    freq, time, spec = spectrogram(s, fs=4096, nperseg=10*4096)

    plt.figure()
    plt.plot(t, s)
    plt.plot(t, hp)
    plt.xlabel('time[s]')
    plt.ylabel('h(t)')

    plt.figure()
    plt.plot(t[:-1], np.diff(phi)/(2.0*np.pi)*fs)
    plt.xlabel('time[sec]')
    plt.ylabel('frequency[Hz]')

    plt.figure()
    plt.pcolormesh(time, freq, spec, cmap='gray')
    plt.colorbar()
    plt.xlabel('time[s]')
    plt.ylabel('frequency[Hz]')


    plt.show()
