import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from pycbc.waveform import get_td_waveform
from pycbc.types.timeseries import TimeSeries




def memory_waveform(return_memory=False, return_as_numpy=False, **kwargs):

    if not 'distance' in kwargs:
        kwargs['distance'] = 1.0

    if not 'inclination' in kwargs:
        kwargs['inclination'] = 0.0


    Mpc = 3.085677581 * 1e+24    #cgs
    c = 2.99792458 * 1e+10    #cgs
    
    # the waveform from the actual source
    hp, hc = get_td_waveform(**kwargs)

    # the waveform for calculation of the memory
    eta = kwargs['inclination']
    dist = kwargs['distance']
    inc_factor = np.sin(eta)**2.0 * (1.0 - np.sin(eta)**2.0 / 18.0)


    kwargs['inclination'] = 0.0
    h, _ = get_td_waveform(**kwargs)
        
    hdot = np.gradient(h, h.delta_t)
    hdot2_int = cumtrapz(hdot**2.0, h.sample_times, initial=0.0)
    
    '''    
    hpdot = np.gradient(hp, hp.delta_t)
    hcdot = np.gradient(hc, hc.delta_t)
    hpdot2_int = cumtrapz(hpdot**2.0, hp.sample_times, initial=0.0)
    hcdot2_int = cumtrapz(hcdot**2.0, hc.sample_times, initial=0.0)
    hdot2_int = hpdot2_int + hcdot2_int
    '''
    
    h_mem = hdot2_int * (dist*Mpc) / (4*np.pi*c) * inc_factor
    h_mem = TimeSeries(h_mem, kwargs['delta_t'])
    h_mem.start_time = hp.start_time

    if return_as_numpy:
        t = np.array(hp.sample_times)
        hp = np.array(hp)
        h_mem = np.array(h_mem)

        if return_memory:
            return t, hp + h_mem, h_mem

        else:
            return t, hp + h_mem

    if return_memory:
        
        return hp + h_mem, h_mem
    
    else:
    
        return hp + h_mem





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
    
    
    hp, hc = get_td_waveform(approximant = 'SEOBNRv4',
                             mass1 = m1,
                             mass2 = m2,
                             spin1z = a1,
                             spin2z = a2,
                             inclination = inc,
                             distance = dist,
                             delta_t = 1.0 / fs,
                             f_lower = fmin)

    
    hp_memory, h_mem = memory_waveform(approximant = 'SEOBNRv4',
                                       mass1 = m1,
                                       mass2 = m2,
                                       spin1z = a1,
                                       spin2z = a2,
                                       distance = dist,
                                       inclination = inc,
                                       delta_t = 1.0 / fs,
                                       f_lower = fmin,
                                       return_memory=True)


    

    plt.figure()
    plt.plot(h_mem.sample_times, h_mem)
    plt.xlabel("time [sec]")
    plt.xlim([-0.2, 0.05])
    plt.ylabel("strain")
    plt.title("memory effect for GW150914")
    plt.grid()


    
    plt.figure()
    plt.plot(hp.sample_times, hp, label='no memory')
    plt.plot(hp.sample_times, hp_memory, label='memory added')
    plt.xlabel("time [sec]")
    plt.xlim([-0.2, 0.05])
    plt.ylabel("strain")
    plt.legend()
    plt.grid()




    plt.figure()
    
    for inc in [np.pi/2.0, 2.856]:

        hp_memory, h_mem = memory_waveform(approximant = 'SEOBNRv4',
                                           mass1 = m1,
                                           mass2 = m2,
                                           spin1z = a1,
                                           spin2z = a2,
                                           distance = dist,
                                           inclination = inc,
                                           delta_t = 1.0 / fs,
                                           f_lower = fmin,
                                           return_memory=True)
        
        plt.plot(h_mem.sample_times, h_mem, label='$\eta$ = %.3f'%inc)

    plt.legend()
    plt.xlabel('time[sec]')
    plt.ylabel('strain')
    plt.xlim([-0.2, 0.05])
    plt.grid()
    plt.show()

