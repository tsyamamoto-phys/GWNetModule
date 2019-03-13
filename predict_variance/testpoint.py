import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description="parse args")
parser.add_argument('--resultdir', type=str)
args = parser.parse_args()


data = np.genfromtxt(args.resultdir+'test_output_4.0.txt').T
m1t = data[0]
m2t = data[1]
m1p = data[2]
m2p = data[3]


plt.figure()
plt.scatter(m1t, m2t, s=5, c=abs(m1p-m1t)/m1t*100)
plt.colorbar()
plt.xlabel('m1')
plt.ylabel('m2')

plt.figure()
plt.scatter(m1t, m2t, s=5, c=abs(m2p-m2t)/m2t*100)
plt.colorbar()
plt.xlabel('m1')
plt.ylabel('m2')


plt.show()


