# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:10:42 2023

@author: nrb50
"""
import matplotlib.pyplot as plt
import numpy as np
from mod0_lab2_P1_sol import ButterFilter, Cheby1Filter, FIR
from mod0_lab2_P2_sol import SinglePulseWaveform
from copy import deepcopy as dcp
plt.close('all')

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
		
		self.mywf = SinglePulseWaveform(pulse_width_s = 10e-6,
										pulse_repetition_interval_s = 1000e-6,
										lfm_excursion_hz = 2e6,
										rf_sampling_frequency_hz = 500e6,
										if_sampling_frequency_hz = 100e6,
										bb_sampling_frequency_hz = 25e6,
										rf_center_frequency_hz = 115e6)
		
	def apply_rf2if_filter(self,x): return self.rf2if_filter.filter_signal(x)
	def apply_adc_filter(self,x): return self.adc_filter.filter_signal(x)
	
	def process_signal(self,x):
		fig,axes = plt.subplots()
		x = self.apply_rf2if_filter(x) #Can be bypassed if you don't have anything out of band.
		x = x[::self.rf2if_ds]
		x = self.apply_adc_filter(x)
		x = x * np.exp(-1j*2*np.pi/self.Fs_if * self.fc_if *np.arange(len(x)))
		x = self.mywf.apply_bb_filter(x)
		x = x[::self.if2bb_ds]
		
		#fig.savefig('../SignalProcessingTutorial/figs/rfchaintest.png')
		x = np.convolve(x,np.conj(self.mywf.mf_wf_bb), mode = 'same')
		axes.plot(np.abs(x))
		fig.savefig('./distance_delay_test.png')
		return x

def ffts(x): return np.fft.fftshift(np.fft.fft(x))/len(x)
def affts(x): return np.abs(ffts(x))



if __name__ == '__main__':
	myreceiver = Receiver()
	
	#Calculate index of signal presence
	d = 50000 #distance of target in meters
	distance_samples_skin_return_m = np.arange(myreceiver.mywf.samples_per_cpi_rf) / myreceiver.Fs_rf * 3e8/2
	print(f'Maximum Distance: {np.max(distance_samples_skin_return_m)}, Target Distance: {d}')
	min_range_sample_to_d = np.argmin(np.abs(distance_samples_skin_return_m-d))
	
	#Truncate return signals outside cpi, and concatenate zeros
	x = dcp(myreceiver.mywf.wf())
	fig,axes = plt.subplots()
	
	x = x[:(myreceiver.mywf.samples_per_cpi_rf-min_range_sample_to_d)]
	x = np.concatenate([np.zeros(myreceiver.mywf.samples_per_cpi_rf-len(x)) + 0.0j,x])
	axes.plot(np.real(x))
	axes.plot(np.imag(x))
	myreceiver.process_signal(x)
