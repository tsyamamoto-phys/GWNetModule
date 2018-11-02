from pycbc.waveform import get_td_waveform
from pycbc.types.timeseries import TimeSeries
import numpy as np
import matplotlib.pyplot as plt


hp, hc = get_td_waveform(approximant='SEOBNRv4',
                         mass1=30.0,
                         mass2=30.0,
                         delta_t = 1.0/4096,
                         f_lower=20.0)

print(len(hp))


h_np = np.array(hp)


h_re = TimeSeries(h_np, delta_t=1.0/4096)
h_re.start_time = hp.start_time

print(h_re.duration, h_re.delta_t)

hp_f = hp.to_frequencyseries()
h_re_f = h_re.to_frequencyseries()

plt.figure()
plt.loglog(hp_f.sample_frequencies, abs(hp_f))
plt.loglog(h_re_f.sample_frequencies, abs(h_re_f))
plt.show()



