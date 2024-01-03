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

class Simulation:
	'''
	Top level simulation class for a 1v1 target vs track radar
	'''
	def __init__(self):
	
		self.target = Scatterer()
								
		self.radar = Radar()
		
			
	def run_sim(self):
		wf_object = self.radar.mywf
		x = self.radar.transmitter.transmit_waveform(wf_object)
		
		#Truth target information
		zoa,aoa,d = self.target.get_scatterer_entity_geo(self.radar.transmitter)
		distance_samples_skin_return_m = np.arange(wf_object.samples_per_cpi_rf) / self.radar.receiver.Fs_rf * 3e8/2
		
		min_range_sample_to_d = np.argmin(np.abs(distance_samples_skin_return_m-d))
		
		#Truncate return signals outside cpi
		x = x[:(wf_object.samples_per_cpi_rf-min_range_sample_to_d)]
		
		x = np.concatenate([np.zeros(wf_object.samples_per_cpi_rf-len(x)) + 0.0j,x])
		
			
		#RRE
		G2 = 10**(30/10) #placeholder gain for antenna transmit and receive
		x = x * np.sqrt( G2 * (3e8/self.radar.transmitter.fc_rf)**2 * self.target.rcs_lin / d**4  / (4*np.pi)**3)
		
		x = self.radar.receiver.process_signal(x,wf_object)
		print(f'Maximum Distance: {np.max(distance_samples_skin_return_m)}, Target Distance: {d}')
		return x

class Radar:
	'''
	Basic single mode, single pulse radar
	'''
	def __init__(self):
		self.transmitter = Transmitter()
		self.receiver = Receiver()
		
		self.mywf = SinglePulseWaveform(pulse_width_s = 10e-6,
										pulse_repetition_interval_s = 1000e-6,
										lfm_excursion_hz = 2e6,
										rf_sampling_frequency_hz = self.receiver.Fs_rf,
										if_sampling_frequency_hz = self.receiver.Fs_if,
										bb_sampling_frequency_hz = self.receiver.Fs_bb,
										rf_center_frequency_hz = self.receiver.fc_rf)
										
class Receiver:
	def __init__(self,
			  
				  #Spatial Parameters
					x_loc_m = 0.0, 
					y_loc_m = 0.0,
					z_loc_m = 3.0, 
					x_vel_mps = 0.0,
					y_vel_mps = 0.0,
					z_vel_mps = 0.0,
					x_acc_mps2 = 0.0,
					y_acc_mps2 = 0.0,
					z_acc_mps2 = 0.0,
					
					
					rf_sampling_frequency_hz = 500e6,
					if_sampling_frequency_hz = 100e6,
					bb_sampling_frequency_hz = 25e6,
					rf_center_frequency_hz = 115e6,
					rf_bandwidth_hz = 10e6,
					
					
					receiver_noise_figure_db = 5,
					
					#####NEW##################
					reference_cells_one_sided = 30,
					guard_cells_one_sided = 5,
					probability_false_alarm = 1e-6,
					detector_type = 'square'
					##########################
					):
		
		self.state = np.array([x_loc_m,y_loc_m,z_loc_m,x_vel_mps,y_vel_mps,z_vel_mps]) 
		self.Fs_rf = rf_sampling_frequency_hz
		self.Fs_if = if_sampling_frequency_hz
		self.Fs_bb = bb_sampling_frequency_hz
		self.fc_rf = rf_center_frequency_hz
		self.fc_if = np.mod(rf_center_frequency_hz,if_sampling_frequency_hz)
		self.rf_bw = rf_bandwidth_hz
		self.det_type = detector_type
		
		self.NF_lin = 10**(receiver_noise_figure_db/10)
		self.sigma_n = np.sqrt(1.38e-23 * 290 * rf_bandwidth_hz * self.NF_lin)
		
		self.rf2if_ds = int(self.Fs_rf/self.Fs_if)
		self.if2bb_ds = int(self.Fs_if/self.Fs_bb)
		
		self.rf2if_filter = ButterFilter(N = 2,Wn = [self.fc_rf - self.rf_bw/2,self.fc_rf + self.rf_bw/2],fs = self.Fs_rf,btype = 'bandpass')
		self.adc_filter = Cheby1Filter(N = 5,rp = 2, Wn = 20e6,fs = self.Fs_if, btype = 'low')
		self.bb_filter = FIR(numtaps = 31, cutoff = 1e6, fs = self.Fs_bb)
		
		#####NEW##################
		self.cfar = CA_CFAR1D(reference_cells_one_sided, guard_cells_one_sided,probability_false_alarm)
		##########################
		
	def add_receiver_noise(self,x): return x + self.sigma_n/np.sqrt(2) * (np.random.randn(len(x)) + 1j*np.random.randn(len(x)))
	
	
	def apply_rf2if_filter(self,x): return self.rf2if_filter.filter_signal(x)
	def apply_adc_filter(self,x): return self.adc_filter.filter_signal(x)
	
	def process_signal(self,x,wf_object):
		x = self.apply_rf2if_filter(x) #Can be bypassed if you don't have anything out of band.
		x = x[::self.rf2if_ds]
		x = self.add_receiver_noise(x)
		x = self.apply_adc_filter(x)
		x = x * np.exp(-1j*2*np.pi/self.Fs_if * self.fc_if *np.arange(len(x)))
		x = wf_object.apply_bb_filter(x)
		x = x[::self.if2bb_ds]
		
		x = np.convolve(x,np.conj(wf_object.mf_wf_bb), mode = 'same')
		#####NEW##################
		x,T = self.detect_single_signal(x)
		##########################
		
		return x,T
	
	#####NEW##################
	def detector(self,x):
		x = np.abs(x)
		if self.det_type == 'square': x = x**2
		return x
		
	def detect_single_signal(self,x):
		x = self.detector(x)
		T = self.cfar.calculate_cfar_thresh(x)
		return x,T
	##########################
class Transmitter:
	def __init__(self,
	
				#Spatial Parameters
				x_loc_m = 0.0, 
				y_loc_m = 0.0,
				z_loc_m = 3.0, 
				x_vel_mps = 0.0,
				y_vel_mps = 0.0,
				z_vel_mps = 0.0,
				x_acc_mps2 = 0.0,
				y_acc_mps2 = 0.0,
				z_acc_mps2 = 0.0,
				
				#Transmitter and Sampling Parameters
				rf_sampling_frequency_hz = 500e6,
				if_sampling_frequency_hz = 100e6,
				bb_sampling_frequency_hz = 25e6,
				rf_center_frequency_hz = 115e6,
				rf_bandwidth_hz = 10e6,
				transmit_power_w = 5000):
				
		self.state = np.array([x_loc_m,y_loc_m,z_loc_m,x_vel_mps,y_vel_mps,z_vel_mps]) 
		self.Fs_rf = rf_sampling_frequency_hz
		self.Fs_if = if_sampling_frequency_hz
		self.Fs_bb = bb_sampling_frequency_hz
		self.fc_rf = rf_center_frequency_hz
		self.fc_if = np.mod(rf_center_frequency_hz,if_sampling_frequency_hz)
		self.rf_bw = rf_bandwidth_hz
		self.Ptx = transmit_power_w

	def transmit_waveform(self,wf_object):
		return np.sqrt(self.Ptx) * wf_object.wf()

class Scatterer:
	def __init__(self,
	
				#Spatial Parameters
				x_loc_m = 50000, 
				y_loc_m = 0.0,
				z_loc_m = 10000, 
				x_vel_mps = 0.0,
				y_vel_mps = 0.0,
				z_vel_mps = 0.0,
				x_acc_mps2 = 0.0,
				y_acc_mps2 = 0.0,
				z_acc_mps2 = 0.0,
				
				#Signature
				radar_cross_section_dbsm = 10):
					
		self.state = np.array([x_loc_m,y_loc_m,z_loc_m,x_vel_mps,y_vel_mps,z_vel_mps]) 
		self.rcs_dbsm = radar_cross_section_dbsm
		self.rcs_lin = 10**(self.rcs_dbsm/10)
			
	def get_scatterer_entity_geo(self,entity):
		''' 
		x, y, and z distance relative to some entity, i.e. a transmitter.
		'''
		x = self.state[0] - entity.state[0]
		y = self.state[1] - entity.state[1]
		z = self.state[2] - entity.state[2]
		zoa = np.arctan(np.sqrt(x**2 + y**2)/z)
		aoa = np.sign(y) * np.arccos(x/np.sqrt(x**2 + y**2))
		d = np.sqrt((x)**2 + (y)**2 + (z)**2)
		return zoa,aoa,d

class CA_CFAR1D:
	def __init__(self,num_reference_cells_one_sided,
				  num_guard_cells_one_sided,
				  probability_of_false_alarm):
		self.num_ref = num_reference_cells_one_sided
		self.num_guard = num_guard_cells_one_sided
		self.pfa = probability_of_false_alarm
		
		N = 2 * num_reference_cells_one_sided
		self.cfar_constant = N * (probability_of_false_alarm**(-1/N) -1)
		self.cfar_window = self.cfar_constant/N * np.concatenate([np.ones(self.num_ref),np.zeros(2*self.num_guard+ 1),np.ones(self.num_ref)])
		
	def calculate_cfar_thresh(self,x):
		return np.convolve(x,self.cfar_window, mode = 'same')
	
	def build_detection_vector(self,x):
		T = self.calculate_cfar_thresh(x)
		det_vec = np.zeros(len(x)).astype('int')
		det_vec[x>T] = 1
		return det_vec
		
def ffts(x): return np.fft.fftshift(np.fft.fft(x))/len(x)
def affts(x): return np.abs(ffts(x))



if __name__ == '__main__':
	mysim = Simulation()
	x,T = mysim.run_sim()
	fig,axes = plt.subplots()
	axes.plot(10*np.log10(x))
	axes.plot(10*np.log10(T))
	fig.savefig('./cfar_test.png')
	
	fig,axes = plt.subplots()
	xbin = mysim.radar.receiver.cfar.build_detection_vector(x)
	axes.plot(xbin)
	fig.savefig('./det_vec.png')