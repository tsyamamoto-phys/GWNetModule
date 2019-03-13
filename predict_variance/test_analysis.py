import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="parse args")
parser.add_argument('--resultdir', type=str)
args = parser.parse_args()

snr_list = np.arange(4.0, 0.5, -0.1)

for snr in snr_list:
    f = args.resultdir + '/test_output_%.1f.txt' % snr
    data = np.genfromtxt(f).T

    m1t = data[0]
    m2t = data[1]
    m1p = data[2]
    m2p = data[3]

    err1 = np.mean(abs(m1p - m1t) / m1t)
    err2 = np.mean(abs(m2p - m2t) / m2t)

    err = (err1 + err2) / 2.0

    print("%.1f %.3f %.3f %.3f" % (snr, err1*100., err2*100., err*100.))

