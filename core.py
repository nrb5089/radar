import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import butter, cheby1, lfilter,firwin, convolve2d
from copy import deepcopy as dcp
import cv2
from util import ffts,affts,log2lin

		
class MonostaticRadar:
	def __init__(self,params):
		self.params = params
		self.m_per_sample_rf =3e8 / 2 / self.params['rf_sampling_frequency_hz']
		self.m_per_sample_if =3e8 / 2 / self.params['if_sampling_frequency_hz']
		self.m_per_sample_bb =3e8 / 2 / self.params['bb_sampling_frequency_hz']
		
		self.transmitter = BasicTransmitter(
								x_loc_m = params['x_loc_m_tx'], 
								y_loc_m = params['y_loc_m_tx'],
								z_loc_m = params['z_loc_m_tx'], 
								x_vel_mps = params['x_vel_mps_tx'],
								y_vel_mps = params['y_vel_mps_tx'],
								z_vel_mps = params['z_vel_mps_tx'],
								x_acc_mps2 = params['x_acc_mps2_tx'],
								y_acc_mps2 = params['y_acc_mps2_tx'],
								z_acc_mps2 = params['z_acc_mps2_tx'],
								rf_sampling_frequency_hz = params['rf_sampling_frequency_hz'],
								if_sampling_frequency_hz = params['if_sampling_frequency_hz'],
								bb_sampling_frequency_hz = params['bb_sampling_frequency_hz'],
								rf_center_frequency_hz = params['rf_center_frequency_hz'],
								rf_bandwidth_hz = params['rf_bandwidth_hz'],
								transmit_power_w = params['transmit_power_w'], #per element
								internal_loss_db = params['internal_loss_db_tx'])
								
		self.receiver = BasicReceiver(
						x_loc_m = params['x_loc_m_rx'], 
						y_loc_m = params['y_loc_m_rx'],
						z_loc_m = params['z_loc_m_rx'], 
						x_vel_mps = params['x_vel_mps_rx'],
						y_vel_mps = params['y_vel_mps_rx'],
						z_vel_mps = params['z_vel_mps_rx'],
						x_acc_mps2 = params['x_acc_mps2_rx'],
						y_acc_mps2 = params['y_acc_mps2_rx'],
						z_acc_mps2 = params['z_acc_mps2_rx'],
						rf_sampling_frequency_hz = params['rf_sampling_frequency_hz'],
						if_sampling_frequency_hz = params['if_sampling_frequency_hz'],
						bb_sampling_frequency_hz = params['bb_sampling_frequency_hz'],
						rf_center_frequency_hz = params['rf_center_frequency_hz'],
						rf_bandwidth_hz = params['rf_bandwidth_hz'],
						internal_loss_db = params['internal_loss_db_rx'],
						num_reference_cells_range_one_sided = params['num_reference_cells_range_one_sided'],
						num_guard_cells_range_one_sided = params['num_guard_cells_range_one_sided'],
						num_reference_cells_doppler_one_sided = params['num_reference_cells_doppler_one_sided'],
						num_guard_cells_doppler_one_sided = params['num_guard_cells_doppler_one_sided'],
						probability_false_alarm = params['probability_false_alarm'],
						probability_false_alarm_2D = params['probability_false_alarm_2D'],
						detector_type = params['detector_type'])
		
		
		#Specify Waveforms by populating 'self.wf_bank', each element is a wf_object
		self.wf_bank = {}
		for wf_desc in params['wf_list']:
			if wf_desc['type'] == 'single': 
				WFClass = SinglePulseWaveform
			if wf_desc['type'] == 'burst':
				WFClass = BurstPulseWaveform
			self.wf_bank[wf_desc['index']] = WFClass(pulse_width_s = wf_desc['pw'],
							pulse_repetition_interval_s = wf_desc['pri'],
							lfm_excursion_hz = wf_desc['lfm_excursion'],
							pris_per_coherent_processing_interval = wf_desc['pris_per_cpi'],
							rf_sampling_frequency_hz = params['rf_sampling_frequency_hz'],
							if_sampling_frequency_hz = params['if_sampling_frequency_hz'],
							bb_sampling_frequency_hz = params['bb_sampling_frequency_hz'],
							rf_center_frequency_hz = params['rf_center_frequency_hz'])

		#Initialize Mode and Waveform
		self.current_mode_index = params['starting_mode_index']
		self.reset_mode()
	
	def waveform_scheduler_cpi(self):
		'''Method called iteratively that correpsonds to defined mode operation to generate waveform'''
		# mode_dict = self.params['wf_sequences'][self.current_mode_index]
		# num_unique_wfs_per_mode = len(self.mode_dict['sequence'])
		wf_index = self.mode_sequence[self.current_mode_waveform_index]
		wf_object = self.wf_bank[wf_index]
		wfout = self.transmitter.transmit_waveform(wf_object)
		
		self.current_mode_waveform_index = np.mod(self.current_mode_waveform_index + 1,self.num_unique_wfs_per_mode)
		return wfout, wf_object
	
	def switch_mode(self,mode_index):
		self.current_mode_index = mode_index
		self.reset_mode()
	
	def reset_mode(self):
		# self.current_mode_waveform_index = self.params['wf_sequences'][self.current_mode_index]['sequence'][0]
		self.current_mode_waveform_index = 0
		self.mode_dict = self.params['wf_sequences'][self.current_mode_index]
		self.mode_sequence = self.mode_dict['sequence']
		
		self.num_unique_wfs_per_mode = len(self.mode_sequence)
		samples_per_pris = []
		self.pris_per_cpi_wfs = []
		self.pris_in_mode_sequence = []
		for wf_index in self.mode_sequence:
			samples_per_pris.append(self.wf_bank[wf_index].samples_per_pri_bb)
			self.pris_per_cpi_wfs.append(self.wf_bank[wf_index].pris_per_cpi)
			self.pris_in_mode_sequence.append(self.wf_bank[wf_index].pri)
		self.min_range_samples_for_display_bb = np.min(samples_per_pris)
		self.min_doppler_samples_for_display_two_sided = np.min(self.pris_per_cpi_wfs)
		if np.mod(self.min_doppler_samples_for_display_two_sided,2) != 0:
			self.min_doppler_samples_for_display_one_sided = int((self.min_doppler_samples_for_display_two_sided-1)/2)
		else:
			self.min_doppler_samples_for_display_one_sided = int((self.min_doppler_samples_for_display_two_sided)/2)
			
		
		
class PlanarAESA:
	'''
	Calculates the gain for a particular set of azimuth and elevation angles
	
	Not intended to be a true time delay model
	'''
	def __init__(self,one_sided_elements, array_element_spacing):
		self.num_elements = int(one_sided_elements**2)
		self.d = array_element_spacing
		self.rvec = self.build_rvec(one_sided_elements,array_element_spacing)
		self.zlims = np.array([-np.pi/4,3*np.pi/4])
		self.azlims = np.array([-np.pi/2,np.pi/2])
		
	def array_response(self,azimth_angle,zenith_angle,frequency):
		lam = 3e8/frequency
		kvec = 2*np.pi/lam * np.array([np.sin(zenith_angle)*np.cos(azimth_angle),np.sin(zenith_angle)*np.sin(azimth_angle),np.cos(zenith_angle)])
		return np.exp(1j*self.rvec @ kvec)
	
	
	def build_rvec(self,M,d):
	
		ms = np.arange(M)
		ry = ms*d - d*(M-1)/2
		rx = -(d*ms - d*(M-1)/2)*np.sqrt(2)/2
		rz = (d*ms - d*(M-1)/2)*np.sqrt(2)/2
		rvec = []
		for x,z in zip(rx,rz):
			current_set = []
			current_set.append(np.repeat(x,M))
			current_set.append(np.repeat(z,M))
			current_set.append(ry)
			#current_set1 = np.vstack(current_set).T
			rvec.append(np.vstack(current_set).T)
		rvec = np.vstack(rvec)
		#need to switch columns 1 and 2 so you get xyz
		rvec_hold = dcp(rvec)
		rvec[:,1] = rvec_hold[:,2]
		rvec[:,2] = rvec_hold[:,1]
		return rvec

class SincAntennaPattern:
	"""
	Antenna pattern object for a sinc(x) (sin(x)/x) pattern
	"""
	def __init__(self,antenna_params):
			self.azimuth_beam_width = antenna_params['azimuth_beam_width']
			self.elevation_beam_width = antenna_params['elevation_beam_width']
			self.peak_antenna_gain_db = antenna_params['peak_antenna_gain_db']
			self.first_side_lobe_down_az_db = antenna_params['first_side_lobe_down_az_db']
			self.first_side_lobe_down_el_db = antenna_params['first_side_lobe_down_el_db']
			self.second_side_lobe_down_az_db = antenna_params['second_side_lobe_down_az_db']
			self.second_side_lobe_down_el_db = antenna_params['second_side_lobe_down_el_db']
			self.back_lobe_down_db = antenna_params['back_lobe_down_db']
			
			#Count the sidelobes for az and el
			self.num_side_lobes_az = 0
			lim = self.azimuth_beam_width/2
			while lim < np.pi:
				lim += self.azimuth_beam_width/2
				self.num_side_lobes_az += 1
			
			self.num_side_lobes_el = 0
			lim = self.elevation_beam_width/2
			while lim < np.pi:
				lim += self.elevation_beam_width/2
				self.num_side_lobes_el += 1
			
			#Determine position of null prior to backlobe
			if np.mod(self.num_side_lobes_az,2) == 1:  
				self.az_backlobe_null = self.num_side_lobes_az * self.azimuth_beam_width/2
			else:  
				self.az_backlobe_null = (self.num_side_lobes_az-1) * self.azimuth_beam_width/2
			
			if np.mod(self.num_side_lobes_el,2) == 1:  
				self.el_backlobe_null = self.num_side_lobes_el * self.elevation_beam_width/2
			else:  
				self.el_backlobe_null = (self.num_side_lobes_el-1) * self.elevation_beam_width/2
			
			#Sidelobe ratios are the same regardless of frequency, so it's best to just use the standard
			#Backlobe value is the same for both az and el
			self.first_side_lobe_val_az = np.abs(np.sinc(1.5))
			self.second_side_lobe_val_az = np.abs(np.sinc(2.5))
			self.back_lobe_val  = np.abs(np.sinc(self.az_backlobe_null + .25 * self.azimuth_beam_width)) #Nail down backlobe
			self.first_side_lobe_val_el = np.abs(np.sinc(1.5))
			self.second_side_lobe_val_el = np.abs(np.sinc(2.5))
	
			
# 	
	def gain_val(self,azimuth_angle, elevation_angle, default = False):
		val =  log2lin(self.peak_antenna_gain_db) * np.abs(np.sinc(2/self.azimuth_beam_width * azimuth_angle) * np.sinc(2/self.elevation_beam_width * elevation_angle))
		# val =   np.abs(np.sinc(2/self.azimuth_beam_width * azimuth_angle) * np.sinc(2/self.elevation_beam_width * elevation_angle))
		if not default:
			
				
			# if np.abs(azimuth_angle) <= self.azimuth_beam_width/2: val *= log2lin(self.peak_antenna_gain_db)
			if np.abs(azimuth_angle) > self.azimuth_beam_width/2 and np.abs(azimuth_angle) <= 2*self.azimuth_beam_width/2: val *= 1/self.first_side_lobe_val_az *log2lin(-self.first_side_lobe_down_az_db)
			elif np.abs(azimuth_angle) > 2*self.azimuth_beam_width/2 and np.abs(azimuth_angle) <= self.az_backlobe_null: val *= 1/self.second_side_lobe_val_az*log2lin(-self.second_side_lobe_down_az_db)
			elif np.abs(azimuth_angle) > self.az_backlobe_null: val *= 1/np.sqrt(self.back_lobe_val) * log2lin(-np.sqrt(self.back_lobe_down_db))
			
			if np.abs(elevation_angle) > self.elevation_beam_width/2 and np.abs(elevation_angle) <= 2*self.elevation_beam_width/2: val *= 1/self.first_side_lobe_val_el*log2lin(-self.first_side_lobe_down_el_db)
			elif np.abs(elevation_angle) > 2*self.elevation_beam_width/2 and np.abs(elevation_angle) <= self.el_backlobe_null: val *= 1/self.second_side_lobe_val_el*log2lin(-self.second_side_lobe_down_el_db)
			elif np.abs(elevation_angle) > self.el_backlobe_null: val *= 1/np.sqrt(self.back_lobe_val) * log2lin(-np.sqrt(self.back_lobe_down_db))
		
		return val
	
	def azimuth_slice(self,azimuth_angles,elevation_angle = 0):
		vals = []
		for azimuth_angle in azimuth_angles:
			vals.append(self.gain_val(azimuth_angle,elevation_angle,default = False))
		return np.array(vals)
		
	def elevation_slice(self,elevation_angles,azimuth_angle = 0):
		vals = []
		for elevation_angle in elevation_angles:
			vals.append(self.gain_val(azimuth_angle,elevation_angle))
		return np.array(vals)  
	
# class ContinuousScan:
	# def __init__(self,params):
		# self.period = params['period']
		# if params['type'] = circular:
			# pass

class DiscreteScan:
	def __init__(self,params):
		self.num_steered_positions = len(params['steered_azimuth_angles_rads'])
		self.steered_azimuth_angles_rads = params['steered_azimuth_angles_rads']
		self.steered_elevation_angles_rads = params['steered_elevation_angles_rads']
		self.current_steered_index = params['initial_steered_index']
		self.antenna_pattern = params['antenna_pattern']
		
	def next_steered_position(self): 
		self.current_steered_index += 1
		if self.current_steered_index == self.next_steered_positions: self.current_steered_index = 0
		
		
class BasicTransmitter:
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
				rf_sampling_frequency_hz = 32e9,
				if_sampling_frequency_hz = 4e9,
				bb_sampling_frequency_hz = 25e6,
				rf_center_frequency_hz = 13.2e9,
				rf_bandwidth_hz = 1e9,
				transmit_power_w = 100, #per element
				internal_loss_db = 2):
				
		self.state = np.array([x_loc_m,y_loc_m,z_loc_m,x_vel_mps,y_vel_mps,z_vel_mps]) 
		self.Fs_rf = rf_sampling_frequency_hz
		self.Fs_if = if_sampling_frequency_hz
		self.Fs_bb = bb_sampling_frequency_hz
		self.fc_rf = rf_center_frequency_hz
		self.fc_if = np.mod(rf_center_frequency_hz,if_sampling_frequency_hz)
		self.rf_bw = rf_bandwidth_hz
		self.Ptx = transmit_power_w
		self.Ltx = 10**(internal_loss_db/10)

	def transmit_waveform(self,wf_object):
		return np.sqrt(self.Ptx) * wf_object.wf()
		
class BasicReceiver:
	'''
	Transmits a sequence of lfm pulses as a pulse-Doppler burst and executes basic MTI and FFT processing
	'''
	#TODO: Break it up into an "ACTUAL" RF center frequency and a "CALCULATIONS" RF Center frequency
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
				
				#Receiver and Sampling Parameters
				rf_sampling_frequency_hz = 32e9,
				if_sampling_frequency_hz = 4e9,
				bb_sampling_frequency_hz = 25e6,
				rf_center_frequency_hz = 13.2e9,
				rf_bandwidth_hz = 1e9,
				receiver_noise_figure_db = 5,
				internal_loss_db = 2,
				
				# CFAR Parameters
				num_reference_cells_range_one_sided = 10,
				num_guard_cells_range_one_sided = 4,
				probability_false_alarm = 1e-6,
				num_reference_cells_doppler_one_sided = 10,
				num_guard_cells_doppler_one_sided = 3,
				probability_false_alarm_2D = 1e-1,
				detector_type = 'square'):
		
		self.state = np.array([x_loc_m,y_loc_m,z_loc_m,x_vel_mps,y_vel_mps,z_vel_mps]) 
		self.Fs_rf = rf_sampling_frequency_hz
		self.Fs_if = if_sampling_frequency_hz
		self.Fs_bb = bb_sampling_frequency_hz
		self.fc_rf = rf_center_frequency_hz
		self.fc_if = np.mod(rf_center_frequency_hz,if_sampling_frequency_hz)
		self.rf_bw = rf_bandwidth_hz
		self.det_type = detector_type
		
		self.rf_lam = 3e8/self.fc_rf
		self.rf2if_ds = int(self.Fs_rf/self.Fs_if)
		self.if2bb_ds = int(self.Fs_if/self.Fs_bb)
		
		self.NF_lin = 10**(receiver_noise_figure_db/10)
		self.Lrx = 10**(internal_loss_db/10)
		
		#TODO Add Power limiting, like a tanh
		
		self.sigma_n = np.sqrt(1.38e-23 * 290 * rf_bandwidth_hz * self.NF_lin)
		
		#Analog Front End Filters
		#assert self.fc_if + self.rf_bw/2 < self.Fs_rf/2, f'Top of RF signal bandwidth {(self.fc_rf + self.rf_bw/2 )/2/1e6} MHz equals or exceeds sampling Nyquist limit {self.Fs_rf/2/1e6} MHz!'
		self.rf2if_filter = ButterFilter(N = 2,Wn = [self.fc_rf - self.rf_bw/2,self.fc_rf + self.rf_bw/2],fs = self.Fs_rf,btype = 'bandpass')
		self.adc_filter = Cheby1Filter(N = 5,rp = 2, Wn = self.Fs_if/2 -10e6,fs = self.Fs_if, btype = 'low')
		
		#Doppler Processing
		self.mti = MTIBasic()
		
		# #Detection
		self.cfar = CA_CFAR1D(num_reference_cells_range_one_sided, num_guard_cells_range_one_sided,probability_false_alarm)
		self.cfar2D = CA_CFAR2D(num_reference_cells_doppler_one_sided, 
								num_reference_cells_range_one_sided,
								num_guard_cells_doppler_one_sided,
								num_guard_cells_range_one_sided,
								probability_false_alarm_2D)
		
	def detector(self,x):
		x = np.abs(x)
		if self.det_type == 'square': x = x**2
		return x
	
	def apply_rf2if_filter(self,x): return self.rf2if_filter.filter_signal(x)
	def apply_adc_filter(self,x): return self.adc_filter.filter_signal(x)
	def apply_bb_filter(self,x,wf_object): 
		'''BB filter is custom to the MF waveform'''
		return wf_object.bb_filter.filter_signal(x)
	
	def add_receiver_noise(self,x): return x + self.sigma_n/np.sqrt(2) * (np.random.randn(len(x)) + 1j*np.random.randn(len(x)))
	
	def apply_gating(self,x,wf_object,center_range_gate_m): 
		width_range_gate_m = wf_object.samples_per_range_window_rf / wf_object.Fs_rf * 3e8/2
		range_gate_min_m = center_range_gate_m - width_range_gate_m/2
		range_gate_max_m = center_range_gate_m + width_range_gate_m/2
		
		range_sample_min = int(2/3e8 * range_gate_min_m * wf_object.Fs_rf)
		range_sample_max = int(2/3e8 * range_gate_max_m * wf_object.Fs_rf)
		range_gate_mask = np.zeros(wf_object.samples_per_cpi_rf) + 0j
		range_gate_mask[range_sample_min:range_sample_max] = 1.0 + 0j
		return x * range_gate_mask

	def process_burst_signal(self,x,wf_object,probe = False):
		#Mask emulates the receive window times so that there aren't samples uring transmission, for example
		x = x * wf_object.rcv_window_mask
		x = self.process_single_signal(x,wf_object,probe = probe,premasked = True)

		x2D = np.reshape(x,[wf_object.pris_per_cpi,wf_object.samples_per_pri_bb])
		del x
		x2D_filt = []
		for row in x2D.T:
			x2D_filt.append(self.mti.filter_signal(row))
		x2D = np.vstack(x2D_filt)
		del x2D_filt
		# x2D = np.fft.fft(x2D)
		x2D = np.fft.fftshift(np.fft.fft(x2D))
		# x2D = np.concatenate([np.zeros([wf_object.num_zeros_to_pad_range_samples_bb,512]) + 0j,x2D])
		return x2D
	
	def process_single_signal(self,x,wf_object,probe = False, premasked = False):
		if not premasked:
			#Mask emulates the receive window times so that there aren't samples uring transmission, for example
			
			#Burst Signals are already masked
			x = x * wf_object.rcv_window_mask
		if probe:
			fig,axes = plt.subplots(3,2)
			freq = np.linspace(-self.Fs_rf/2,self.Fs_rf/2,len(x))
			axes[0,0].plot(freq/1e6,affts(x))
			axes[0,0].set_xlabel('MHz')
			axes[0,1].plot(np.real(x))
			axes[0,1].plot(np.imag(x))
		x = self.apply_rf2if_filter(x) #Can be bypassed if you don't have anything out of band.
		x = x[::self.rf2if_ds]
		if probe:
			freq = np.linspace(-self.Fs_if/2,self.Fs_if/2,len(x))
			axes[1,0].plot(freq/1e6,affts(x),'b')
			axes[1,0].set_xlabel('MHz')
		x = self.add_receiver_noise(x)
		x = self.apply_adc_filter(x)
		#x = self.add_receiver_noise(x)
		
		if probe:
			axes[1,1].plot(np.real(x))
			axes[1,1].plot(np.imag(x))
		
		x = x * np.exp(-1j*2*np.pi/self.Fs_if * self.fc_if *np.arange(len(x)))
		
		if probe:
			axes[1,0].plot(freq/1e6,affts(x),'r')
		#TODO add fine frequency translation of specific agility channel
		x = self.apply_bb_filter(x,wf_object)
		x = x[::self.if2bb_ds]
		
		if probe:
			freq = np.linspace(-self.Fs_bb/2,self.Fs_bb/2,len(x))
			axes[2,0].plot(freq/1e6,affts(x))
			axes[2,0].set_xlabel('MHz')
			axes[2,1].plot(np.real(x))
			axes[2,1].plot(np.imag(x))
			
		x = wf_object.apply_matched_filter(x)
		
		if probe:
			fig,axes = plt.subplots()
			axes.plot(np.abs(x))
		return x
		
	def detect_single_signal(self,x):
		x = self.detector(x)
		T = self.cfar.calculate_cfar_thresh(x)
		return x,T

	def detect_burst_signal(self,x):
		x = self.detector(x)
		#TODO: Add this.
		T = self.cfar2D.calculate_cfar_thresh(x)
		return x, T
		

class BurstPulseWaveform:
	def __init__(self,
					pulse_width_s = 1e-6,
					pulse_repetition_interval_s = 19e-6,
					lfm_excursion_hz = 10e6,
					pris_per_coherent_processing_interval = 32,
					rf_sampling_frequency_hz = 32e9,
					if_sampling_frequency_hz = 4e9,
					bb_sampling_frequency_hz = 25e6,
					rf_center_frequency_hz = 13.2e9
					):
		
		self.pw = pulse_width_s
		self.pri = pulse_repetition_interval_s
		self.lfm_ex = lfm_excursion_hz
		self.pris_per_cpi = pris_per_coherent_processing_interval
		self.Fs_rf = rf_sampling_frequency_hz
		self.Fs_if = if_sampling_frequency_hz
		self.Fs_bb = bb_sampling_frequency_hz
		self.fc_rf = rf_center_frequency_hz
		self.fc_if = np.mod(rf_center_frequency_hz,if_sampling_frequency_hz)
		
		self.type = 'burst'
		
		self.samples_per_pw_rf = int(self.pw * rf_sampling_frequency_hz)
		self.samples_per_pw_if = int(self.pw * if_sampling_frequency_hz)
		self.samples_per_pw_bb = int(self.pw * bb_sampling_frequency_hz)
		
		self.samples_per_pri_rf = int(self.pri * rf_sampling_frequency_hz)
		self.samples_per_pri_if = int(self.pri * if_sampling_frequency_hz)
		self.samples_per_pri_bb = int(self.pri * bb_sampling_frequency_hz)
		
		self.samples_per_cpi_rf = int(pris_per_coherent_processing_interval * self.samples_per_pri_rf)
		self.samples_per_cpi_if = int(pris_per_coherent_processing_interval * self.samples_per_pri_if)
		self.samples_per_cpi_bb = int(pris_per_coherent_processing_interval * self.samples_per_pri_bb)
		self.cpi_duration_s = self.pris_per_cpi * self.pri
		
		self.samples_per_range_window_rf = self.samples_per_pri_rf - int(2 * self.samples_per_pw_rf)
		self.samples_per_range_window_if = self.samples_per_pri_if - int(2 * self.samples_per_pw_if)
		self.samples_per_range_window_bb = self.samples_per_pri_bb - int(2 * self.samples_per_pw_bb)
		
		self.Delta_R = 3e8/2/self.lfm_ex
		self.samples_per_range_bin_rf = int(self.Fs_rf/self.lfm_ex)
		self.samples_per_range_bin_bb = int(self.Fs_bb/self.lfm_ex)
		self.range_per_cpi_m = self.samples_per_cpi_rf / self.Fs_rf * 3e8/2
		self.range_unambiguous = 3e8 * self.pri/2
		self.fmin_bb = -self.lfm_ex/2
		self.fmax_bb = self.lfm_ex/2
		
		#Digital Decimation Pre-Filter for MF
		self.bb_filter = FIR(numtaps = 31, cutoff = self.fmax_bb, fs = self.Fs_bb)
		
		self.mf_wf_bb = np.exp(1j * 2 * np.pi/self.Fs_bb * (np.cumsum(np.linspace(self.fmin_bb,self.fmax_bb,self.samples_per_pw_bb))))
	
		#Receive window for a PRI starts on the falling edge of the pulse and has a listening time equal to a PRI - 2PW
		self.rcv_window_mask = []
		
		for pri_idx in np.arange(self.pris_per_cpi):
			self.rcv_window_mask.extend(np.zeros(self.samples_per_pw_rf) + 0j)
			self.rcv_window_mask.extend(np.ones(self.samples_per_range_window_rf) + 0j)
			self.rcv_window_mask.extend(np.zeros(int(self.samples_per_pw_rf)) + 0j)
		self.rcv_window_mask = np.array(self.rcv_window_mask)
			
	def wf(self):
		'''
		waveforms set as generators to preserve memoryv
		'''
		wf_single_pri = np.concatenate([np.exp(1j * 2 * np.pi/self.Fs_rf * (self.fc_rf *np.arange(self.samples_per_pw_rf) + np.cumsum(np.linspace(self.fmin_bb,self.fmax_bb,self.samples_per_pw_rf)))),np.zeros(self.samples_per_pri_rf-self.samples_per_pw_rf) + 0j])
		burst_wf = []
		for ii in np.arange(self.pris_per_cpi):
			burst_wf.extend(wf_single_pri)
		return np.array(burst_wf)
		
	def apply_matched_filter(self,x): return np.convolve(x,np.conj(self.mf_wf_bb), mode = 'same')
	def apply_bb_filter(self,x): return self.bb_filter.filter_signal(x)
	
class SinglePulseWaveform:
	def __init__(self,
					pulse_width_s = 10e-6,
					pulse_repetition_interval_s = 1000e-6,
					lfm_excursion_hz = 2e6,
					pris_per_coherent_processing_interval = 1,
					rf_sampling_frequency_hz = 32e9,
					if_sampling_frequency_hz = 4e9,
					bb_sampling_frequency_hz = 25e6,
					rf_center_frequency_hz = 13.2e9
					):
		
		self.pw = pulse_width_s
		self.pri = pulse_repetition_interval_s
		self.lfm_ex = lfm_excursion_hz
		self.pris_per_cpi = pris_per_coherent_processing_interval
		self.Fs_rf = rf_sampling_frequency_hz
		self.Fs_if = if_sampling_frequency_hz
		self.Fs_bb = bb_sampling_frequency_hz
		self.fc_rf = rf_center_frequency_hz
		self.fc_if = np.mod(rf_center_frequency_hz,if_sampling_frequency_hz)
		
		self.type = 'single'
		
		
		self.samples_per_pw_rf = int(self.pw * rf_sampling_frequency_hz)
		self.samples_per_pw_if = int(self.pw * if_sampling_frequency_hz)
		self.samples_per_pw_bb = int(self.pw * bb_sampling_frequency_hz)
		
		self.samples_per_pri_rf = int(self.pri * rf_sampling_frequency_hz)
		self.samples_per_pri_if = int(self.pri * if_sampling_frequency_hz)
		self.samples_per_pri_bb = int(self.pri * bb_sampling_frequency_hz)
		
		self.samples_per_cpi_rf = int(1 * self.samples_per_pri_rf)
		self.samples_per_cpi_if = int(1 * self.samples_per_pri_if)
		self.samples_per_cpi_bb = int(1 * self.samples_per_pri_bb)
		self.cpi_duration_s = self.pri
		
		self.samples_per_range_window_rf = self.samples_per_pri_rf - self.samples_per_pw_rf
		self.samples_per_range_window_if = self.samples_per_pri_if - self.samples_per_pw_if
		self.samples_per_range_window_bb = self.samples_per_pri_bb - self.samples_per_pw_bb
		
		self.Delta_R = 3e8/2/self.lfm_ex
		self.samples_per_range_bin_rf = int(self.Fs_rf/self.lfm_ex)
		self.range_per_cpi_m = self.samples_per_cpi_rf / self.Fs_rf * 3e8/2
		self.fmin_bb = -self.lfm_ex/2
		self.fmax_bb = self.lfm_ex/2
		
		#Digital Decimation Pre-Filter for MF
		self.bb_filter = FIR(numtaps = 31, cutoff = self.fmax_bb, fs = self.Fs_bb)
		
		self.wf_single_pw_rf = np.exp(1j * 2 * np.pi/self.Fs_rf * (self.fc_rf *np.arange(self.samples_per_pw_rf) + np.cumsum(np.linspace(self.fmin_bb,self.fmax_bb,self.samples_per_pw_rf))))
		self.mf_wf_bb = np.exp(1j * 2 * np.pi/self.Fs_bb * (np.cumsum(np.linspace(self.fmin_bb,self.fmax_bb,self.samples_per_pw_bb))))

		#Receive window for a PRI starts on the falling edge of the pulse and has a listening time equal to a PRI - 2PW
		self.rcv_window_mask = []
		self.rcv_window_mask.extend(np.zeros(self.samples_per_pw_rf) + 0j)
		self.rcv_window_mask.extend(np.ones(self.samples_per_range_window_rf) + 0j)
		self.rcv_window_mask = np.array(self.rcv_window_mask)
		
	def wf(self):
		'''
		waveforms set as generators to preserve memoryv
		'''
		if self.pri > 0: return np.concatenate([self.wf_single_pw_rf,np.zeros(self.samples_per_pri_rf-self.samples_per_pw_rf) + 0j])
		else: return self.wf_single_pw_rf
		
		
	# def apply_matched_filter(self,x): return np.convolve(x,np.conj(self.mf_wf_bb), mode = 'full')[self.samples_per_pw_bb:-self.samples_per_pw_bb]
	def apply_matched_filter(self,x): return np.convolve(x,np.conj(self.mf_wf_bb), mode = 'same')
	# def apply_matched_filter(self,x): return np.convolve(x,np.conj(self.mf_wf_bb), mode = 'full')
	def apply_bb_filter(self,x): return self.bb_filter.filter_signal(x)


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
	
	def build_detection_vector(self,x,T):
		#T = self.calculate_cfar_thresh(x)
		det_vec = np.zeros(len(x)).astype('int')
		det_vec[x>T] = 1
		return det_vec

class CA_CFAR2D:
	def __init__(self,num_reference_cells_doppler_one_sided,
				num_reference_cells_range_one_sided,
				num_guard_cells_doppler_one_sided,
				num_guard_cells_range_one_sided,
				probability_false_alarm):
		self.num_ref_d = num_reference_cells_doppler_one_sided
		self.num_ref_r = num_reference_cells_range_one_sided
		self.num_guard_d = num_guard_cells_doppler_one_sided
		self.num_guard_r = num_guard_cells_range_one_sided
		self.pfa = probability_false_alarm
		
		N = int(num_reference_cells_doppler_one_sided * num_reference_cells_range_one_sided - num_guard_cells_doppler_one_sided*num_guard_cells_range_one_sided)
		self.cfar_constant = N * (probability_false_alarm**(-1/N) -1)
		self.cfar_window = self.cfar_constant/N * np.ones([2*(num_reference_cells_range_one_sided + num_guard_cells_range_one_sided)+ 1,2*(num_reference_cells_doppler_one_sided + num_guard_cells_doppler_one_sided)+1])
		self.cfar_window[num_reference_cells_range_one_sided:num_reference_cells_range_one_sided + 2*num_guard_cells_range_one_sided+1,num_reference_cells_doppler_one_sided:num_reference_cells_doppler_one_sided + 2*num_guard_cells_doppler_one_sided + 1] = 0.0
	
	def calculate_cfar_thresh(self,x):
		return convolve2d(x,self.cfar_window,mode = 'same')
		
	def build_detection_matrix(self,x,T):
		#T = self.calculate_cfar_thresh(x)
		det_mat = np.zeros(x.shape).astype('int')
		det_mat[x>T] = 1
		return det_mat

class MTIBasic:
	def __init__(self):
		self.h = np.array([1,-1]) * 1/2
		
	def filter_signal(self,x): return np.convolve(x,self.h,mode = 'same')
	
class CoherentOscillator:
	def __init__(self, frequency_hz, initial_phase):
		self.current_phase = initial_phase

if __name__ == '__main__':
	init_figs()
	plt.close('all')
# 	myradar = Receiver(rf_sampling_frequency_hz = 500e6,
# 					if_sampling_frequency_hz = 100e6,
# 					bb_sampling_frequency_hz = 20e6,
# 					rf_center_frequency_hz = 20e6,
# 					rf_bandwidth_hz = 10e6)

	single_pulse_demo()
	#test_path()
	#test_array()
	#demo_doppler_maps()
	#tracking_sim()
	#myradar = Receiver()
	#x = myradar.pd_wf_object.wf
	#x = myradar.process_signal(x)
	#x = myradar.process_probe_signal(x,myradar.pd_wf_object)
	#x = myradar.test()
	plt.show()
	


