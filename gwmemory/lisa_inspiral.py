import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import tukey
from scipy.constants import G, c
from scipy.special import gamma, fresnel

PI = np.pi
PI2 = PI*PI
Msun = 1.9884 * 1e+30
print(G, c)

def inspiral(t, f, Mc, amp=1.0, phi0=0.0):

    
    df = (96* PI**(8.0/3.0) / 5) * f**(11.0/3.0) * (Mc*Msun)**(5.0/3.0) * (G/c**3.0)**(5.0/3.0)
    print(df)
    #amp = 5.0 * df / (96 * PI2 * f**3.0 * dist) / c
    phi = 2*PI*(f+0.5*df*t)*t + phi0
    hp = amp * np.cos(phi)
    hc = -amp * np.sin(phi)

    return hp, hc


def integralsin2(x):

    fresnelc, fresnels = fresnel(np.sqrt(2./PI)*x)
    intc = 3*x*np.sin(x**2.0)*gamma(3./4.) / (8.*gamma(7./4.)) - 3.0*np.sqrt(2.0*PI)*fresnels*gamma(3./4.) / (16.*gamma(7./4.))
    ints = -5*x*np.cos(x**2.)*gamma(5./4.)/(8.*gamma(9./4.)) + 5.*np.sqrt(2.*PI)*fresnelc*gamma(5./4.)/(16.*gamma(9./4.))

    return intc, ints


def inspiral_memory(t, f, Mc, amp=1.0, phi0=0.0):

    df = (96* PI**(8.0/3.0) / 5) * f**(11.0/3.0) * (Mc*Msun)**(5.0/3.0) * (G/c**3.0)**(5.0/3.0)

    intc = integralsin2(np.sqrt(2*PI*df)*(t+f/df))[0] - integralsin2(np.sqrt(2*PI*df)*f/df)[0]
    ints = integralsin2(np.sqrt(2*PI*df)*(t+f/df))[1] - integralsin2(np.sqrt(2*PI*df)*f/df)[1]
    h = 2*PI2*(amp**2.0) * (df*(t**3.0)/3.0 + f*df*(t**2.0) + f*t)\
        - np.sqrt(df/2./PI)/(2.*PI) * (np.cos(2*PI*(f**2.)/df)*intc + np.sin(2*PI*(f**2.)/df)*ints) / 2.0 
    
    return h




time = np.arange(0.0, 1e+7, 1.0/3.0)
h = inspiral_memory(time, 1e-3, 1.0)

plt.figure()
plt.plot(time, h)
plt.show()
