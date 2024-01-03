import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy as dcp
import cv2
from core import MonostaticRadar, PlanarAESA

class Simulation:
	'''
	Top level simulation class for a 1v1 target vs track radar
	'''
	def __init__(self, sim_params, target_params, radar_params, demo = False):
		
		self.sim_params = sim_params
		self.target_params = target_params
		self.radar_params = radar_params
		
		self.target = DWNATarget(target_params)
								
		self.radar = MonostaticRadar(radar_params)
		
		self.process_rf = sim_params['process_rf'] 
		self.process_lam = 3e8/self.process_rf
		self.radar.aesa = PlanarAESA(32, 3e8/2/self.process_rf)
			
	# def process_cpi(self,wf_object,steered_az,steered_z,x, noenv = False):
		
		# #Truth target informaitn
		# zoa,aoa,d = self.target.get_target_entity_geo(self.radar.transmitter)
		# distance_samples_skin_return_m = np.arange(wf_object.samples_per_cpi_rf) / self.radar.receiver.Fs_rf * 3e8/2
		
		# min_range_sample_to_d = np.argmin(np.abs(distance_samples_skin_return_m-d))
		
		# #Truncate return signals outside cpi
		# x = x[:(wf_object.samples_per_cpi_rf-min_range_sample_to_d)]
		
		# x = np.concatenate([np.zeros(wf_object.samples_per_cpi_rf-len(x)) + 0.0j,x])
		
		# if noenv: return x
		# else:
			# #Doppler
			# #TODO Doppler v_d seems like it may be always positive, need to fix this
			# v_zoa,v_aoa,v_d = self.target.get_target_entity_relative_velocity(self.radar.transmitter)
			# x = x * np.exp(1j* 2*np.pi/self.radar.receiver.Fs_rf * 2* v_d/self.process_lam * np.arange(len(x))) #TODO Should there be phase variation in this?
			
			# #RRE
			# x = x * np.sqrt(self.process_lam**2 * self.target.rcs / d**4  / (4*np.pi)**3 / self.radar.receiver.Lrx / self.radar.transmitter.Ltx)
			
			# #Tx/Rx Beamforming 
			# # x = np.abs(np.conj(self.radar_aesa.array_response(steered_az,steered_z,self.process_rf)) @ self.radar_aesa.array_response(aoa,zoa,self.process_rf) )**2 * x
			# x = np.abs(np.conj(self.radar.aesa.array_response(steered_az,steered_z,self.process_rf)) @ self.radar.aesa.array_response(aoa,zoa,self.process_rf) ) * x
			
			# print(f'Maximum Distance: {np.max(distance_samples_skin_return_m)}, Target Distance: {d}, Target Radial Velocity: {v_d}')
			# return x

	# def process_dwell(self):
		
	def process_mode_cpi(self,steered_az,steered_z, noenv = False, probe = False):
		x, wf_object = self.radar.waveform_scheduler_cpi()
		#Truth target informaitn
		zoa,aoa,d = self.target.get_target_entity_geo(self.radar.transmitter)
		d_t = 2/3e8 * d
		distance_samples_skin_return_t = np.arange(wf_object.samples_per_cpi_rf) / self.radar.receiver.Fs_rf 
		
		min_range_sample_to_t = np.argmin(np.abs(distance_samples_skin_return_t-d_t))
		
		#Truncate return signals outside cpi
		x = np.concatenate([np.zeros(min_range_sample_to_t) + 0.0j,x])
		x = x[:wf_object.samples_per_cpi_rf]
		
		
		if noenv: 
			pass
		else:
			#Doppler
			#TODO Doppler v_d seems like it may be always positive, need to fix this
			v_zoa,v_aoa,v_d = self.target.get_target_entity_relative_velocity(self.radar.transmitter)
			x = x * np.exp(1j* 2*np.pi/self.radar.receiver.Fs_rf * 2* v_d/self.process_lam * np.arange(len(x))) #TODO Should there be phase variation in this?
			
			#RRE
			x = x * np.sqrt(self.process_lam**2 * self.target.rcs / d**4  / (4*np.pi)**3 / self.radar.receiver.Lrx / self.radar.transmitter.Ltx)
			
			#Tx/Rx Beamforming 
			# x = np.abs(np.conj(self.radar_aesa.array_response(steered_az,steered_z,self.process_rf)) @ self.radar_aesa.array_response(aoa,zoa,self.process_rf) )**2 * x
			x = np.abs(np.conj(self.radar.aesa.array_response(steered_az,steered_z,self.process_rf)) @ self.radar.aesa.array_response(aoa,zoa,self.process_rf) ) * x
			
# 			print(f'Maximum Distance: {np.max(distance_samples_skin_return_m)}, Target Distance: {d}, Target Radial Velocity: {v_d}')
			print(f'Maximum Distance: {np.max(distance_samples_skin_return_t*3e8/2)}, Target Distance: {d}, Target Radial Velocity: {v_d}')
		
		if False: x = self.radar.receiver.apply_gating(x,wf_object,d)
		if wf_object.type == 'single': 
			x = self.radar.receiver.process_single_signal(x,wf_object)
		
		if wf_object.type == 'burst': 
			#x is 2D now
			x = self.radar.receiver.process_burst_signal(x,wf_object,probe = probe)
			
			#RDM Display processing
			h,w = x.shape
			halfw = int(np.floor(w/2))
			x = x[:self.radar.min_range_samples_for_display_bb,int(halfw-self.radar.min_doppler_samples_for_display_one_sided):int(halfw+self.radar.min_doppler_samples_for_display_one_sided)]
			
			x,T = self.radar.receiver.detect_burst_signal(x)
			B = self.radar.receiver.cfar2D.build_detection_matrix(x,T)
		#Update Target State	
		self.target.update_state(wf_object.cpi_duration_s)
		
		
		return x,B
		
class DWNATarget:
	'''
	Moves under a DWNA model
	'''
	def __init__(self,target_params):
		x_loc_m = target_params['x_loc_m'] #100 nmi
		y_loc_m = target_params['y_loc_m']
		z_loc_m = target_params['z_loc_m'] #35kft
		x_vel_mps = target_params['x_vel_mps'] #550 knots Remember this is relative to the radarr
		y_vel_mps = target_params['y_vel_mps']
		z_vel_mps = target_params['z_vel_mps']
		x_acc_mps2 = target_params['x_acc_mps2']
		y_acc_mps2 = target_params['y_acc_mps2']
		z_acc_mps2 = target_params['z_acc_mps2']
		radar_cross_section_dbsm = target_params['radar_cross_section_dbsm']

		self.state = np.array([x_loc_m,y_loc_m,z_loc_m,x_vel_mps,y_vel_mps,z_vel_mps])
		self.sigma_accs = np.array([x_acc_mps2,y_acc_mps2,z_acc_mps2])
		self.rcs = 10**(radar_cross_section_dbsm/10)
		
	def update_state(self,time_between_last_measurement):
		t = time_between_last_measurement
		ax = self.sigma_accs[0] * np.random.randn(1)[0]
		ay = self.sigma_accs[1] * np.random.randn(1)[0]
		az = self.sigma_accs[2] * np.random.randn(1)[0]
		self.state[0] = self.state[0] + t*self.state[3] + t**2/2 * ax
		self.state[1] = self.state[1] + t*self.state[4] + t**2/2 * ay
		self.state[2] = self.state[2] + t*self.state[5] + t**2/2 * az
		self.state[3] = self.state[3] + t*ax
		self.state[4] = self.state[4] + t*ay
		self.state[5] = self.state[5] + t*az
	
	def get_target_entity_geo(self,entity):
		x = self.state[0] - entity.state[0]
		y = self.state[1] - entity.state[1]
		z = self.state[2] - entity.state[2]
		zoa = np.arctan(np.sqrt(x**2 + y**2)/z)
		aoa = np.sign(y) * np.arccos(x/np.sqrt(x**2 + y**2))
		d = np.sqrt((x)**2 + (y)**2 + (z)**2)
		return zoa,aoa,d

	def get_target_entity_relative_velocity(self,entity):
		x = self.state[3] - entity.state[3]
		y = self.state[4] - entity.state[4]
		z = self.state[5] - entity.state[5]
		if z == 0: v_zoa = 0.0
		else: v_zoa = np.arctan(np.sqrt(x**2 + y**2)/z)
		v_aoa = np.sign(y) * np.arccos(x/np.sqrt(x**2 + y**2))
		v_d = np.sqrt((x)**2 + (y)**2 + (z)**2)
		return v_zoa,v_aoa,v_d


