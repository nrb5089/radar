# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:10:42 2023

@author: nrb50
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, cheby1, firwin,lfilter

class ButterFilter:
	def __init__(self,N,Wn,fs,btype):
		self.N = N
		self.Wn = Wn
		self.Fs = fs
		self.btype = btype
		
		self.b,self.a = butter(N = N, Wn = Wn, fs = fs, btype = btype)
	
	def filter_signal(self,x): return lfilter(self.b,self.a,x)

class Cheby1Filter:
	def __init__(self,N,rp,Wn,fs,btype):
		self.N = N
		self.rp = rp
		self.Wn = Wn
		self.Fs = fs
		self.btype = btype
		
		self.b,self.a = cheby1(N = N, rp = rp, Wn = Wn, fs = fs, btype = btype)
	
	def filter_signal(self,x): return lfilter(self.b,self.a,x)
	
class FIR:
	def __init__(self,numtaps, cutoff, fs):
		self.numtaps = numtaps
		self.cutoff = cutoff
		self.Fs = fs
		
		self.h = firwin(numtaps = numtaps, cutoff = cutoff, fs = fs)
	
	def filter_signal(self,x): return np.convolve(x,self.h,mode = 'same')
	
class Receiver:
	def __init__(self,
				  rf_sampling_frequency_hz = 500e6,
					if_sampling_frequency_hz = 100e6,
					bb_sampling_frequency_hz = 25e6,
					rf_center_frequency_hz = 115e6,
					rf_bandwidth_hz = 10e6):
		
		self.Fs_rf = rf_sampling_frequency_hz
		self.Fs_if = if_sampling_frequency_hz
		self.Fs_bb = bb_sampling_frequency_hz
		self.fc_rf = rf_center_frequency_hz
		self.fc_if = np.mod(rf_center_frequency_hz,if_sampling_frequency_hz)
		self.rf_bw = rf_bandwidth_hz
		
		
		self.rf2if_ds = int(self.Fs_rf/self.Fs_if)
		self.if2bb_ds = int(self.Fs_if/self.Fs_bb)
		
		self.rf2if_filter = ButterFilter(N = 2,Wn = [self.fc_rf - self.rf_bw/2,self.fc_rf + self.rf_bw/2],fs = self.Fs_rf,btype = 'bandpass')
		self.adc_filter = Cheby1Filter(N = 5,rp = 2, Wn = 20e6,fs = self.Fs_if, btype = 'low')
		self.bb_filter = FIR(numtaps = 31, cutoff = 1e6, fs = self.Fs_bb)

	def apply_rf2if_filter(self,x): return self.rf2if_filter.filter_signal(x)
	def apply_adc_filter(self,x): return self.adc_filter.filter_signal(x)
	def apply_bb_filter(self,x): return self.bb_filter.filter_signal(x)
	
	def process_signal(self,x):
		### FOR PLOT PROCESSING ONLY ##########################
		fig,axes = plt.subplots(3,2)
		freq = np.linspace(-self.Fs_rf/2,self.Fs_rf/2,len(x))
		axes[0,0].plot(freq/1e6,affts(x))
		axes[0,0].set_xlabel('MHz')
		axes[0,1].plot(np.real(x))
		axes[0,1].plot(np.imag(x))
		#######################################################
		
		x = self.apply_rf2if_filter(x) #Can be bypassed if you don't have anything out of band.
		x = x[::self.rf2if_ds]
		
		### FOR PLOT PROCESSING ONLY ##########################
		freq = np.linspace(-self.Fs_if/2,self.Fs_if/2,len(x))
		axes[1,0].plot(freq/1e6,affts(x),'b')
		axes[1,0].set_xlabel('MHz')
		#######################################################
		
		x = self.apply_adc_filter(x)
		
		### FOR PLOT PROCESSING ONLY ##########################
		axes[1,1].plot(np.real(x))
		axes[1,1].plot(np.imag(x))
		#######################################################
		
		x = x * np.exp(-1j*2*np.pi/self.Fs_if * self.fc_if *np.arange(len(x)))
		
		### FOR PLOT PROCESSING ONLY ##########################
		axes[1,0].plot(freq/1e6,affts(x),'r')
		#######################################################
		
		x = self.apply_bb_filter(x)
		x = x[::self.if2bb_ds]
		
		### FOR PLOT PROCESSING ONLY ##########################
		freq = np.linspace(-self.Fs_bb/2,self.Fs_bb/2,len(x))
		axes[2,0].plot(freq/1e6,affts(x))
		axes[2,0].set_xlabel('MHz')
		axes[2,1].plot(np.real(x))
		axes[2,1].plot(np.imag(x))
		#######################################################
		#fig.savefig('../SignalProcessingTutorial/figs/rfchaintest.png')
		fig.savefig('./rfchaintest.png')
		return x

def ffts(x): return np.fft.fftshift(np.fft.fft(x))/len(x)
def affts(x): return np.abs(ffts(x))

if __name__ == '__main__':
	pulse_width = 10e-6
	Fs_rf = 500e6
	fc_rf = 115e6
	lfm_min = -1e6
	lfm_max = 1e6
	signal_length_samples = int(pulse_width * Fs_rf) #5000
	x = np.exp(1j * 2 * np.pi/Fs_rf * (fc_rf *np.arange(signal_length_samples) + np.cumsum(np.linspace(lfm_min,lfm_max,signal_length_samples))))
	
	myreceiver = Receiver()
	myreceiver.process_signal(x)
