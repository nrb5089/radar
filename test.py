# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 22:10:39 2023

@author: nrb50
"""

from main import Cheby1Filter, affts, FIR
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
plt.close('all')
fc_if = 15e6
rf_bw = 10e6
Fs_if = 100e6

Fs_bb = 25e6
fmax_bb = 1e6
#mycheby = Cheby1Filter(N = 5,rp = 2, Wn = fc_if + rf_bw/2,fs = Fs_if, btype = 'low')
mycheby = Cheby1Filter(N = 5,rp = 2, Wn = Fs_if/2-10e6,fs = Fs_if, btype = 'low')
myfir = FIR(numtaps = 31, cutoff = fmax_bb, fs = Fs_bb)

plt.figure(0)
w, h = signal.freqz(mycheby.b, mycheby.a)
plt.semilogx(w*Fs_if/2/np.pi, 20 * np.log10(abs(h)))

plt.figure(1)
H = affts(myfir.h)
freq = np.linspace(-Fs_bb/2,Fs_bb/2,len(H))
plt.plot(freq/1e6,10*np.log10(H))