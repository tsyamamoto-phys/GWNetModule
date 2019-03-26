# based on Jaranowski, Klorak and Schutz (1998)


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from astropy import units as u
from astropy import constants as const

G = const.G.cgs.value
c = const.c.cgs.value
Mpc = const.pc.cgs.value * (1e+06)
R_E = const.R_earth.cgs.value
R_ES = const.au.cgs.value

class GWDetector():

    def __init__(self, detector):

        self.det = {}

        if detector == 'L1':
            self.det['gamma'] = 243.0 * 2.0 * np.pi / 360.
            self.det['lmd'] = 30.56 * 2.0 * np.pi / 360.
            self.det['L'] = -89.23 * 2.0 * np.pi / 360.

        elif detector == 'H1':
            self.det['gamma'] = 171.8 * 2.0 * np.pi / 360.
            self.det['lmd'] = 46.45 * 2.0 * np.pi / 360.
            self.det['L'] =  -60.59 * 2.0 * np.pi / 360.


        epsilon = 0.3
        coseps = np.cos(epsilon)
        sineps = np.sin(epsilon)
        self.equator_matrix = np.array([[1., 0., 0.],
                                        [0., coseps, sineps],
                                        [0., -sineps, coseps]])



    def target_source(self, time, alpha, delta, pol, phir):

        self.alpha = alpha
        self.delta = delta
        self.pol = pol
        self.spin_phase = phir + 2*np.pi*time/(24*60*60)


    def antenna(self):

        cos2g = np.cos(2.0*self.det['gamma'])
        sin2g = np.sin(2.0*self.det['gamma'])
        coslmd = np.cos(self.det['lmd'])
        sinlmd = np.sin(self.det['lmd'])
        cos2lmd = np.cos(2.0*self.det['lmd'])
        sin2lmd = np.sin(2.0*self.det['lmd'])

        cosa = np.cos(self.alpha - self.spin_phase)
        sina = np.sin(self.alpha - self.spin_phase)
        cos2a = np.cos(2.0*(self.alpha - self.spin_phase))
        sin2a = np.sin(2.0*(self.alpha - self.spin_phase))

        cosdlt = np.cos(self.delta)
        sindlt = np.sin(self.delta)
        cos2dlt = np.cos(2.0*self.delta)
        sin2dlt = np.sin(2.0*self.delta)
        
        antenna_a = sin2g * (3.-cos2lmd) * (3. - cos2dlt) * cos2a / 16.\
            - cos2g * sinlmd * (3. - cos2dlt) * sin2a / 4.\
            + sin2g * sin2lmd * sin2dlt * cosa / 4.\
            - cos2g * coslmd * sin2dlt * cosa / 2.\
            +3. * sin2g * (coslmd**2.0) * (cosdlt**2.0) /4.

        antenna_b = cos2g * sinlmd * sindlt * cos2a\
            + sin2g * (3. - cos2lmd) * sindlt * sin2a / 4.\
            + cos2g * coslmd * cosdlt * cosa\
            + sin2g * sin2lmd * cosdlt * sina / 2.


        return antenna_a, antenna_b



    def detector_loc(self, time):

        phio = 0.0
        orbit_phase = phio + 2.0*np.pi*time/(3.15e+7)
        earth_loc = R_ES * np.array([np.cos(orbit_phase), np.sin(orbit_phase), 0.0])
      
        spin_phase = phir + 2*np.pi*time/(24*60*60)
        det_loc_on_earth = np.array([np.cos(self.det['lmd'])*np.cos(spin_phase),
                                     np.cos(self.det['lmd'])*np.sin(spin_phase),
                                     np.sin(self.det['lmd'])])

        det_loc = R_E * self.equator_matrix.dot(det_loc_on_earth)

        return earth_loc + det_loc


    def line_of_sight(self):

        cosa = np.cos(self.alpha)
        sina = np.sin(self.alpha)
        cosdlt = np.cos(self.delta)
        sindlt = np.sin(self.delta)

        v = np.array([cosa * cosdlt, sina * cosdlt, sindlt])
        return self.equator_matrix.dot(v)


    def phase_modulate_factor(self, time):

        r_d = self.detector_loc(time)
        n_0 = self.line_of_sight()
        return np.dot(r_d, n_0)




class LowMassBinaryGW():

    def __init__(self, time, f, df, alpha, delta, pol, phir, detector):

        self.time = time
        self.f = f
        self.df = df
        self.det = detector
        self.det.target_source(time, alpha, delta, pol, phir)

    def phase(self, phi0):
        
        modulate = self.det.phase_modulate_factor(self.time)
        return phi0 + 2*np.pi*self.f*self.time + np.pi*self.df*(self.time**2.0) + 2.0*np.pi*modulate*self.f / c

    def get_waveform(self, dist, phi0):
        
        amp = 5.0 * c * self.df / (24*np.pi*dist*Mpc*(self.f**3.0))
        phi = self.phase(phi0)

        return amp * np.cos(phi), amp*np.cos(2.0*np.pi*self.time*self.f + np.pi*self.df*(self.time**2.0))





if __name__ == '__main__':

    detector = 'L1'
    alpha = 0.0
    delta = 0.0
    pol = 0.0
    phir = 0.0
    phi0 = 0.0

    time = np.arange(0.0, 60*60*24, 1.0/4096)

    det = GWDetector(detector)
    
    binary = LowMassBinaryGW(time, 100.0, 1e-3, alpha, delta, pol, phir, det)
    h, h2 = binary.get_waveform(400.0, phi0)
    f, t, Zxx = signal.stft(h, 4096, nperseg=4096*60)
    h_f = np.fft.rfft(h)
    freq = np.fft.rfftfreq(len(h), time[1]-time[0])

    def subsample(data,sample_size):
        samples = list(zip(*[iter(data)]*sample_size))
        return map(lambda x: sum(x) / float(len(x)), samples)


    plt.figure()
    plt.plot(time[:4096], h[:4096])


    plt.figure()
    plt.loglog(freq, abs(h_f))
    plt.xlabel('frequency[Hz]')
    plt.ylabel('strain')

    plt.figure()
    plt.pcolormesh(t, f, np.abs(Zxx))
    plt.xlabel('time[sec]')
    plt.ylabel('frequency[Hz]')

    plt.show()

    """
    t = np.arange(0.0, 60*60, 1.0/4096)

    F_plus = antenna_a(t, detector, alpha, delta, phir) * np.cos(2.*pol) \
        + antenna_b(t, detector, alpha, delta, phir) * np.sin(2.*pol)

    F_cross = antenna_b(t, detector, alpha, delta, phir) * np.cos(2.*pol) \
        - antenna_a(t, detector, alpha, delta, phir) * np.sin(2.*pol)
    
    Phi = 2.0*np.pi*t*100.0

    hp = np.cos(Phi)
    hc = np.sin(Phi)

    h = hp * F_plus + hc * F_cross

    plt.figure()
    plt.plot(t, F_plus)
    plt.plot(t, F_cross)
    plt.xlabel('time[s]')
    plt.ylabel('F')
    
    plt.figure()
    plt.plot(t, h)
    plt.xlabel('time[s]')
    plt.ylabel('h')
    
    plt.show()

    """
