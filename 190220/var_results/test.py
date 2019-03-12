import numpy as np
import matplotlib.pyplot as plt


data = np.genfromtxt('test_outputs_4.0.txt').T
m1t = data[0]
m2t = data[1]
m1p = data[2]
m2p = data[3]


nlist = [0, 100]
for n in nlist:
    Lambda = np.zeros((2,2))
    Lambda[0,0] = data[4,n]
    Lambda[0,1] = data[6,n]
    Lambda[1,0] = data[6,n]
    Lambda[1,1] = data[5,n]
    Sigma = np.linalg.inv(Lambda)
    print(Sigma)

    print((m1p-m1t)[n], (m2p-m2t)[n])
    print(np.sqrt(Sigma[0,0]), np.sqrt(Sigma[1,1]))


"""
plt.figure()
plt.scatter((m1p-m1t)/m1t*100, (m2p-m2t)/m2t*100, s=3)
plt.xlabel('relative error in m1 [%]')
plt.ylabel('relative error in m2 [%]')
plt.show()
"""

