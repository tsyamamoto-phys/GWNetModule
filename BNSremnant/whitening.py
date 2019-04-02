import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)

    # whitening: transform to freq domain, divide by asd, then transform back, 
    # taking care to get normalization right.
    hf = np.fft.rfft(strain)
    norm = 1./np.sqrt(1./(dt*2))
    white_hf = hf / interp_psd(freqs) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht


if __name__ == '__main__':

    from pycbc.waveform import get_td_waveform

    psddata = np.genfromtxt('ZERO_DET_high_P.txt')
    f = psddata[:,0]
    psd = psddata[:,1]

    interp_psd = interp1d(f, psd, fill_value="extrapolate")

    hp, hc = get_td_waveform(approximant='SEOBNRv4',
                             mass1=10.0, mass2=10.0,
                             f_lower=20.0,
                             delta_t=1.0/4096)

    t = np.array(hp.sample_times)
    dt = t[1] - t[0]
    hp = np.array(hp)

    hp_wh = whiten(hp, interp_psd, dt)

    plt.figure()
    plt.plot(t, hp_wh)
    plt.show()
