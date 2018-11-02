import pycbc.catalog, pylab
import matplotlib.pyplot as plt

m = pycbc.catalog.Merger("GW150914")
ts = m.strain("L1").time_slice(m.time - 8, m.time +8)

white = ts.whiten(4,4)
hp = white.highpass_fir(35.0, 8)
bp = hp.lowpass_fir(300.0, 8)

plt.figure()
plt.plot(bp.sample_times, bp)
plt.show()
