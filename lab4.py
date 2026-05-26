import numpy as np 
import matplotlib.pyplot as plt 
N=100 
mean = -10 
sigma = 4 
massive_normal = np.random. normal(mean, sigma, N)

t = np.linspace(0, 2*np.pi, N)
ampl1=3 
f1=4
massive_sin_1=ampl1*np.sin(f1*t)

ampl2=3 
f2=2
massive_sin_2=ampl2*np.sin(f2*t)

ampl3=4
f3=5
massive_sin_3=ampl3*np.sin(f3*t)
signal_sum=massive_sin_1+massive_sin_2+massive_sin_3
additive_mix= signal_sum+massive_normal

plt.figure(1)
plt.plot(massive_normal)
plt.title('noise')
plt.grid("on")

plt.figure(2)
plt.plot(massive_normal)
plt.title('sum')
plt.grid("on")

plt.figure(3)
plt.plot(additive_mix)
plt.title('additive mix signal and noise')
plt.grid("on")

plt.show()