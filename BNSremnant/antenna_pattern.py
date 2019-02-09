import numpy as np
import matplotlib.pyplot as plt


def Detectors(detectors):
    if detectors == 'L1':
        gamma = 243.0 * 2.0 * np.pi / 360.
        lmd = 30.56 * 2.0 * np.pi / 360.
        L = 90.77 * 2.0 * np.pi / 360.

    elif detectors=='H1':
        gamma = 171.8 * 2.0 * np.pi / 360.
        lmd = 30.56 * 2.0 * np.pi / 360.
        L = 90.77 * 2.0 * np.pi / 360.
        

    return gamma, lmd, L




def antenna_a(time, alpha, delta, phir=0.0):
    '''
    Parameters of LIGO Livingston [rad]
    '''
    gamma = 243.0 * 2.0 * np.pi / 360.
    lmd = 30.56 * 2.0 * np.pi / 360.
    L = 90.77 * 2.0 * np.pi / 360.
    phi = phir + 2*np.pi*time/(3.154e+7)
    #phi = 0.0

    return np.sin(2.*gamma)*(3.-np.cos(2.*lmd))\
        *(3.-np.cos(2.*delta))*np.cos(2.*(alpha-phi))/16.\
        -np.cos(2.*gamma)*np.sin(lmd)\
        *(3.-np.cos(2.*delta))*np.sin(2.*(alpha-phi))/4.\
        +np.sin(2.*gamma)*np.sin(2.*lmd)*np.sin(2.*delta)*np.cos(alpha-phi)/4.\
        -np.cos(2.*gamma)*np.cos(lmd)*np.sin(2.*delta)*np.cos(alpha-phi)/2.\
        +3.*np.sin(2.*gamma)*(np.cos(lmd)**2.)*(np.cos(delta)**2.)/4.


def antenna_b(time, alpha, delta, phir=0.0):
    '''
    Parameters of LIGO Livingston [rad]
    '''
    gamma = 243.0 * 2.0 * np.pi / 360.
    lmd = 30.56 * 2.0 * np.pi / 360.
    L = 90.77 * 2.0 * np.pi / 360.
    phi = phir + 2*np.pi*time/(3.154e+7)
    #phi = 0.0

    return np.cos(2.*gamma)*np.sin(lmd)*np.sin(delta)*np.cos(2.*(alpha-phi))\
        +np.sin(2.*gamma)*(3.-np.cos(2.*lmd))\
        *np.sin(delta)*np.sin(2.*(alpha-phi))/4.\
        +np.cos(2.*gamma)*np.cos(lmd)*np.cos(delta)*np.cos(alpha-phi)\
        +np.sin(2.*gamma)*np.sin(2.*lmd)*np.cos(delta)*np.sin(alpha-phi)/2.


if __name__ == '__main__':
    from pycbc.waveform import get_td_waveform
    from pycbc.detector import Detector

    det_l1 = Detector('L1')
    
    
    hp, hc = get_td_waveform(approximant='SEOBNRv4',
                             mass1=30., mass2=30.,
                             inclination=0.0,
                             delta_t=1.0/4096,
                             f_lower=20.0)


    pol = 0.0
    alpha = 0.0
    delta = 0.0

    signal_l1 = det_l1.project_wave(hp, hc,
                                    alpha,
                                    delta,
                                    pol)

    t = np.array(hp.sample_times)
    hp = np.array(hp)
    hc = np.array(hc)

    F_plus = antenna_a(t, alpha, delta) * np.cos(2.*pol) \
        + antenna_b(t, alpha, delta) * np.sin(2.*pol)

    F_cross = antenna_b(t, alpha, delta) * np.cos(2.*pol) \
        - antenna_a(t, alpha, delta) * np.sin(2.*pol)

    h = hp * F_plus + hc * F_cross

    t_shift = t[0] - signal_l1.sample_times[0]
    print(t_shift)
    
    plt.figure()
    plt.plot(t, h, label='mine')
    plt.plot(signal_l1.sample_times+0.02, signal_l1, label='PyCBC')
    plt.legend()
    plt.xlabel('time[s]')
    plt.ylabel('h(t)')
    plt.show()
