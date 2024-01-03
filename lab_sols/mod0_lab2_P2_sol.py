# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:10:42 2023

@author: nrb50
"""
import matplotlib.pyplot as plt
import numpy as np
from mod0_lab2_P1_sol import ButterFilter, Cheby1Filter, FIR

class SinglePulseWaveform:
	def __init__(self,
					pulse_width_s = 10e-6,
					pulse_repetition_interval_s = 1000e-6,
					lfm_excursion_hz = 2e6,
					pris_per_coherent_processing_interval = 1,
					rf_sampling_frequency_hz = 500e6,
					if_sampling_frequency_hz = 100e6,
					bb_sampling_frequency_hz = 25e6,
					rf_center_frequency_hz = 115e6,
					):
		
		self.pw = pulse_width_s
		self.pri = pulse_repetition_interval_s
		self.lfm_ex = lfm_excursion_hz
		self.Fs_rf = rf_sampling_frequency_hz
		self.Fs_if = if_sampling_frequency_hz
		self.Fs_bb = bb_sampling_frequency_hz
		self.fc_rf = rf_center_frequency_hz
		self.fc_if = np.mod(rf_center_frequency_hz,if_sampling_frequency_hz)
		
		self.samples_per_pw_rf = int(self.pw * rf_sampling_frequency_hz)
		self.samples_per_pw_if = int(self.pw * if_sampling_frequency_hz)
		self.samples_per_pw_bb = int(self.pw * bb_sampling_frequency_hz)
		
		self.samples_per_pri_rf = int(self.pri * rf_sampling_frequency_hz)
		self.samples_per_pri_if = int(self.pri * if_sampling_frequency_hz)
		self.samples_per_pri_bb = int(self.pri * bb_sampling_frequency_hz)
		
		self.samples_per_cpi_rf = int(1 * self.samples_per_pri_rf)
		self.samples_per_cpi_if = int(1 * self.samples_per_pri_if)
		self.samples_per_cpi_bb = int(1 * self.samples_per_pri_bb)
		
		self.samples_per_range_window_rf = self.samples_per_pri_rf - self.samples_per_pw_rf
		self.samples_per_range_window_if = self.samples_per_pri_if - self.samples_per_pw_if
		self.samples_per_range_window_bb = self.samples_per_pri_bb - self.samples_per_pw_bb
		
		self.Delta_R = 3e8/2/self.lfm_ex
		self.samples_per_range_bin_rf = int(self.Fs_rf/self.lfm_ex)
		self.fmin_bb = -self.lfm_ex/2
		self.fmax_bb = self.lfm_ex/2
		
		#Digital Decimation Pre-Filter for MF
		self.bb_filter = FIR(numtaps = 31, cutoff = self.fmax_bb, fs = self.Fs_bb)
		
		self.wf_single_pw_rf = np.exp(1j * 2 * np.pi/self.Fs_rf * (self.fc_rf *np.arange(self.samples_per_pw_rf) + np.cumsum(np.linspace(self.fmin_bb,self.fmax_bb,self.samples_per_pw_rf))))
		self.mf_wf_bb = np.exp(1j * 2 * np.pi/self.Fs_bb * (np.cumsum(np.linspace(self.fmin_bb,self.fmax_bb,self.samples_per_pw_bb))))
	
	def wf(self):
		'''
		waveforms set as generators to preserve memoryv
		'''
		if self.pri > 0: return np.concatenate([self.wf_single_pw_rf,np.zeros(self.samples_per_pri_rf-self.samples_per_pw_rf) + 0j])
		else: return self.wf_single_pw_rf
		
		
	def apply_matched_filter(self,x): return np.convolve(x,np.conj(self.mf_wf_bb), mode = 'same')
	def apply_bb_filter(self,x): return self.bb_filter.filter_signal(x)
	
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
										pulse_repetition_interval_s = 0.0,
										lfm_excursion_hz = 2e6,rf_sampling_frequency_hz = 500e6,
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
		fig.savefig('./mftest.png')
		return x

def ffts(x): return np.fft.fftshift(np.fft.fft(x))/len(x)
def affts(x): return np.abs(ffts(x))

if __name__ == '__main__':
	myreceiver = Receiver()
	myreceiver.process_signal(myreceiver.mywf.wf_single_pw_rf)
