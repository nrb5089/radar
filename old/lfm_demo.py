

import numpy as np
import matplotlib.pyplot as plt

Fs = 30e9
fc = 10e9
fmin = -10e6
fmax = 10e6
pw = 1e-6
samples_p_pw = int(Fs*pw)

wf = np.exp(1j * 2*np.pi/Fs * (fc + np.cumsum(np.linspace(fmin,fmax,samples_p_pw))))

mfout = np.convolve(wf,np.conj(wf),mode = 'full')

fig,ax = plt.subplots()
ax.plot(np.abs(mfout))
plt.show()