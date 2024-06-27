import numpy as np
from core import Waveform 
import sys
sys.path.insert('../')


class MonostaticRadar:
	def __init__(self,params):
		self.params = params
		self.rx_params = params['rx_params']
		self.tx_params = params['tx_params']
		self.m_per_sample_rf =3e8 / 2 / self.params['rf_sampling_frequency_hz']
		self.m_per_sample_if =3e8 / 2 / self.params['if_sampling_frequency_hz']
		self.m_per_sample_bb =3e8 / 2 / self.params['bb_sampling_frequency_hz']
		
		self.transmitter = BasicTransmitter(params)

		self.receiver = BasicReceiver(params)
		
		
		#Specify Waveforms by populating 'self.wf_bank', each element is a wf_object
		self.wf_bank = {}
		for wf_params in params['wf_list']:
			self.wf_bank[wf_params['index']] = Waveform(params,wf_params)
			
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