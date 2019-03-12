import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('190309MSE/test_outputs_4.0.txt').T
df1_t = abs(data[2] - data[0])
df2_t = abs(data[3] - data[1])
df1_p = np.sqrt(data[4])
df2_p = np.sqrt(data[5])

count = 0
for n in range(data.shape[1]):
    if (df1_t[n]<df1_p[n]) and (df2_t[n]<df2_p[n]):
        count += 1

print("fraction:{0}[%]".format(count/data.shape[1]*100))


plt.figure()
plt.scatter(data[0], data[1], s=5, c=df1_t)

plt.figure()
plt.scatter(data[0], data[1], s=5, c=df2_t)

plt.show()


