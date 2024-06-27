import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy as dcp
import cv2
from sim import Simulation
from util import ffts, affts, init_figs


'''
Other Params
600 kHz PRF, .45us PW, Unmod, 2400 pulses
250 kHz PRF, .75us PW, 13 bit barker, 1000 pulses
200 kHz PRF, 1us PW, .5 MHz LFM, 800 pusles

'''
def main():
	init_figs()
	radar_params = {}
	tx_params['x_loc_m'] = 0.0
	tx_params['y_loc_m'] = 0.0
	tx_params['z_loc_m'] = 3.0
	tx_params['x_vel_mps'] = 0.0
	tx_params['y_vel_mps'] = 0.0
	tx_params['z_vel_mps'] = 0.0
	tx_params['x_acc_mps2'] = 0.0
	tx_params['y_acc_mps2'] = 0.0
	tx_params['z_acc_mps2'] = 0.0
	tx_params['internal_loss_db'] = 2
	
	rx_params['x_loc_m'] = 0.0
	rx_params['y_loc_m'] = 0.0
	rx_params['z_loc_m'] = 3.0
	rx_params['x_vel_mps'] = 0.0
	rx_params['y_vel_mps'] = 0.0
	rx_params['z_vel_mps'] = 0.0
	rx_params['x_acc_mps2'] = 0.0
	rx_params['y_acc_mps2'] = 0.0
	rx_params['z_acc_mps2'] = 0.0
	rx_params['internal_loss_db'] = 2
	
	radar_params['rf_sampling_frequency_hz'] = 500e6
	radar_params['if_sampling_frequency_hz'] = 100e6
	radar_params['bb_sampling_frequency_hz'] = 50e6
	radar_params['rf_center_frequency_hz'] = 115e6
	radar_params['rf_bandwidth_hz'] = 20e6
	radar_params['transmit_power_w'] = 100 #per element
	
	radar_params['tx_params'] = tx_params
	radar_params['rx_params'] = rx_params
									
	detector_params['num_reference_cells_range_one_sided'] = 20
	detector_params['num_guard_cells_range_one_sided'] = 7
	detector_params['num_reference_cells_doppler_one_sided'] = 10
	detector_params['num_guard_cells_doppler_one_sided'] = 3
	detector_params['probability_false_alarm'] = 1e-3
	detector_params['probability_false_alarm_2D'] = 1e-2
	detector_params['detector_type'] = 'square'
	
	num_pris = 7 * 11*13/np.array([7,11,13])
	#num_pris = 7*11*13/np.array([7,7,7])
	num_pris = num_pris.astype('int')
	
	#wf_list is a list of wf_params for each waveform the radar exhibits
	radar_params['wf_list'] = [#{'index': 0, 'type': 'single', 'pw': 100e-6, 'pri': 1500e-6, 'modulation' : 'lfm', 'lfm_excursion' : 2e6, 'bpsk_seq' : [], 'bpsk_chipw' : 0.,'pris_per_cpi': 1},
							#{'index': 1, 'type': 'single', 'pw': 100e-6, 'pri': 1550e-6, 'modulation' : 'lfm', 'lfm_excursion' : 2e6, 'bpsk_seq' : [], 'bpsk_chipw' : 0.,'pris_per_cpi': 1},
							#{'index': 2, 'type': 'single', 'pw': 100e-6, 'pri': 1100e-6, 'modulation' : 'lfm', 'lfm_excursion' : 2e6, 'bpsk_seq' : [], 'bpsk_chipw' : 0.,'pris_per_cpi': 1},
								#{'index' : 3,'type': 'burst', 'pw': 1e-06, 'pri': 7e-6, 'modulation' : 'lfm', 'lfm_excursion' : 2e6, 'bpsk_seq' : [], 'bpsk_chipw' : 0.,'pris_per_cpi': num_pris[0]},
								#{'index' :4,'type': 'burst', 'pw': 1e-06, 'pri': 11e-6, 'modulation' : 'lfm', 'lfm_excursion' : 2e6, 'bpsk_seq' : [], 'bpsk_chipw' : 0.,'pris_per_cpi': num_pris[1]},
								{'index' : 5,'type': 'burst', 'pw': .75e-06, 'pri': 4e-6, 'modulation' : 'bpsk', 'lfm_excursion' : 0., 'bpsk_seq' : [1,1,1,1,1,-1,-1,1,1,-1,1,-1,1], 'bpsk_chipw' : .04e-6,'pris_per_cpi': 200}]
								#{'index' : 6,'type': 'burst', 'pw': .45e-06, 'pri': 1.66667e-6, 'modulation' : 'none', 'lfm_excursion' : 0., 'bpsk_seq' : [], 'bpsk_chipw' : 0.,'pris_per_cpi': 2400},
								#{'index' : 7,'type': 'burst', 'pw': 1e-06, 'pri': 5e-6, 'modulation' : 'lfm', 'lfm_excursion' : 5e6, 'bpsk_seq' : [],'bpsk_chipw' : 0.,'pris_per_cpi': 800}]
								#{'index' : 6,'type': 'burst', 'pw': 1e-06, 'pri': 17e-6, 'lfm_excursion' : 2e6,'pris_per_cpi': num_pris[3]}]
	
	radar_params['wf_sequences'] = [{'index': 0, 'type' : 'single_pulse_stagger', 'sequence' : [0]},
						  {'index': 1, 'type' : 'single_pulse_stagger', 'sequence' : [0,1,2,0,1,2,1]},
						  {'index' : 2, 'type' : 'track','sequence' : [5]},
						  {'index' : 3, 'type' : 'range_resolve', 'sequence' : [3,4]},
							{'index' : 4, 'type' : 'range_resolve', 'sequence' : [3,4,5]},
							{'index' : 5, 'type' : 'range_resolve', 'sequence' : [3,4,5,6]}]
	
	radar_params['starting_mode_index'] = 2
	
	antenna_params = {}
	antenna_params['azimuth_beam_width'] = 15 * np.pi/180
	antenna_params['elevation_beam_width'] = 25 * np.pi/180
	antenna_params['peak_antenna_gain_db'] = 0
	antenna_params['first_side_lobe_down_az_db'] = 10
	antenna_params['first_side_lobe_down_el_db'] = 8
	antenna_params['second_side_lobe_down_az_db'] = 15
	antenna_params['second_side_lobe_down_el_db'] = 12
	antenna_params['back_lobe_down_db'] = 20
	
									
	target_params = {}
	target_params['x_loc_m'] = 31000.0 #100 nmi
	target_params['y_loc_m'] = 0
	target_params['z_loc_m'] = 10668 #35kft
	target_params['x_vel_mps'] = -550 #550 knots Remember this is relative to the radar
# 	target_params['x_vel_mps'] = -500 #550 knots Remember this is relative to the radar
	target_params['y_vel_mps'] = 0
	target_params['z_vel_mps'] = 0.0
	target_params['x_acc_mps2'] = .1
	target_params['y_acc_mps2'] = .001
	target_params['z_acc_mps2'] = .001
	target_params['radar_cross_section_dbsm'] =35
	
	sim_params = {}
	#sim_params['process_rf'] = radar_params['rf_center_frequency_hz']
	sim_params['process_rf'] = 10e9
	
	mysim = Simulation(sim_params,target_params,radar_params,antenna_params)
	#single_pulse_stagger_single_dwell_test()
	#single_pulse_stagger_single_dwell_test()
	#test_path(mysim)
	single_pulse_demo(mysim)
	#mysim.radar.wf_bank[5].help()
	#locs = test_path(mysim)
	#test_array()
	#visualize_doppler_pulses(mysim)
	#demo_doppler_maps(mysim)
	#tracking_sim(mysim)
	#multi_prf_burst_detections(mysim)
	#demo_360_scan(mysim)
	#myradar = Receiver()
	#x = myradar.pd_wf_object.wf
	#x = myradar.process_signal(x)
	#x = myradar.process_probe_signal(x,myradar.pd_wf_object)
	#x = myradar.test()
	plt.show()
	
	
def test_build():
	return
	
def single_pulse_demo(mysim):
	zoa,aoa,d = mysim.target.get_target_entity_geo(mysim.radar.transmitter)
	steered_az = aoa
	steered_z = zoa
	x = mysim.radar.transmitter.transmit_waveform(mysim.radar.wf_bank[0])
	x = mysim.process_cpi(mysim.radar.wf_bank[0],steered_az,steered_z,x,noenv = False)
	x = mysim.radar.receiver.process_single_signal(x,mysim.radar.wf_bank[0], probe = True)
	x,T = mysim.radar.receiver.detect_single_signal(x)
	fig,axes = plt.subplots()
	axes.plot(10*np.log10(x))
	axes.plot(10*np.log10(T))
	
def test_path(mysim):
	locs = []
	Dt = 1
	ts = np.arange(0,250,Dt)
	for t in ts:
		locs.append(dcp(mysim.target.state))
		mysim.target.update_state(Dt)
	max_dev = mysim.target.sigma_accs * len(ts)*Dt
	locsx = [locst[0] for locst in locs]
	locsy = [locst[1] for locst in locs]
	locsz = [locst[2] for locst in locs]
	fig,ax = plt.subplots(subplot_kw={'projection': '3d'})
	ax.plot(locsx,locsy,locsz, marker = '.')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	#ax.set_xlim([locsx[0]-10000,locsx[0]+10000])
	ax.set_ylim([locsy[0]-100,locsy[0]+100])
	ax.set_zlim([0,15000])
	return locs
	

def visualize_doppler_pulses(mysim):
	fig0,axes0 = plt.subplots(mysim.radar.num_unique_wfs_per_mode,1,sharex = True)
	fig1,axes1 = plt.subplots(mysim.radar.num_unique_wfs_per_mode,1,sharex = True)
	overall_mask = 0.0
	for ii in np.arange(mysim.radar.num_unique_wfs_per_mode):
		zoa,aoa,d = mysim.target.get_target_entity_geo(mysim.radar.transmitter)
		steered_az = aoa
		steered_z = zoa
		x, wf_object = mysim.radar.waveform_scheduler_cpi()
		axes0[ii].plot(np.abs(x))
		#Truth target informaitn
		d_t = 2/3e8 * d
		distance_samples_skin_return_t = np.arange(wf_object.samples_per_cpi_rf) / mysim.radar.receiver.Fs_rf 
		
		min_range_sample_to_t = np.argmin(np.abs(distance_samples_skin_return_t-d_t))
		
		#Received pulse burst in each time window
		x = np.concatenate([np.zeros(min_range_sample_to_t) + 0.0j,x])
		x = x[:wf_object.samples_per_cpi_rf]
		axes1[ii].plot(np.abs(x),alpha = .3, color = 'k')
		x = x * wf_object.rcv_window_mask
		overall_mask += wf_object.rcv_window_mask
		axes1[ii].plot(wf_object.rcv_window_mask)
		axes1[ii].plot(np.abs(x))
		mysim.target.update_state(wf_object.cpi_duration_s)
	fig2,axes2 = plt.subplots()
	overall_mask[overall_mask>=1] = 1
	axes2.plot(overall_mask)
	return
	
def demo_doppler_maps(mysim):
	x = mysim.radar.transmitter.transmit_waveform(mysim.radar.wf_bank[1])
	
	num_frames = 2
	
	RDmaps = []
	for ii in np.arange(num_frames):
		if ii ==0: probe = True
		else: probe = False
		if np.mod(ii,10) == 0:
			fig,ax = plt.subplots(1,10)
		zoa,aoa,d = mysim.target.get_target_entity_geo(mysim.radar.transmitter)
		steered_az = aoa
		steered_z = zoa
		x = mysim.process_mode_cpi(steered_az,steered_z,probe = probe) 
		RDmaps.append(x)
		try: ax[int(np.mod(ii,10))].imshow(10*np.log10(np.abs(x)))
		except: ax.imshow(10*np.log10(np.abs(x)))
		
def multi_prf_burst_detections(mysim):
	num_cycles = 2
	fig0,ax0 = plt.subplots(1,num_cycles*mysim.radar.num_unique_wfs_per_mode)
	fig1,ax1 = plt.subplots(1,num_cycles*mysim.radar.num_unique_wfs_per_mode)
	data = np.zeros([mysim.radar.min_range_samples_for_display_bb,2*mysim.radar.min_doppler_samples_for_display_one_sided]) + 0j
	det_maps = []
	for pri_idx in np.arange(num_cycles*mysim.radar.num_unique_wfs_per_mode):
		zoa,aoa,d = mysim.target.get_target_entity_geo(mysim.radar.transmitter)
		steered_az = aoa
		steered_z = zoa
		x,B = mysim.process_mode_cpi(steered_az,steered_z)
		data += x
		det_maps.append(B)
		ax0[pri_idx].imshow(B)
		ax1[pri_idx].imshow(20*np.log10(np.abs(x)))
	print('pause')
		

def demo_360_scan(mysim):
	xs,Bs = mysim.perform_360_scan(.25)
	fig1,ax1 = plt.subplots(1,1)
	fig2,ax2 = plt.subplots(1,1)
	xs = np.vstack(xs)
	Bs = np.vstack(Bs)
	ax1.imshow(np.abs(xs))
	ax1.imshow(Bs)
	return 
	
def single_pulse_stagger_single_dwell_test(mysim):
	wf_index_sequence = mysim.radar.params['wf_sequences'][mysim.radar.current_mode_index]['sequence']
	zoa,aoa,d = mysim.target.get_target_entity_geo(mysim.radar.transmitter)
	steered_az = aoa
	steered_z = zoa
	pris = []
	pws = []
	for wf_index in wf_index_sequence:
		print(mysim.radar.wf_bank[wf_index].pri)
		pris.append(mysim.radar.wf_bank[wf_index].pri)
		pws.append(mysim.radar.wf_bank[wf_index].pw)
	num_cpis_per_dwell = len(wf_index_sequence)
	fig,axes = plt.subplots(num_cpis_per_dwell,1,sharex = True)
	for ii in np.arange(num_cpis_per_dwell):
		x = mysim.process_mode_cpi(steered_az, steered_z)
		x,T = mysim.radar.receiver.detect_single_signal(x)
		mysim.target.update_state(pris[ii])
		#Need to shift by half a pulsewidth since the spike will be in the middle of the pulse
		distance_for_plot = (np.linspace(0,pris[ii],len(x)) - pws[ii]/2) * 3e8/2
		try:
			axes[ii].plot(distance_for_plot,10*np.log10(x))
			axes[ii].plot(distance_for_plot,10*np.log10(T))
		except:
			axes.plot(distance_for_plot,10*np.log10(x))
			axes.plot(distance_for_plot,10*np.log10(T))
		
def tracking_sim(mysim):


	# Initialize video window
	window_name = 'Range Doppler Display'
	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

	# Initialize video writer
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	#video_writer = cv2.VideoWriter('output_video1.mp4', fourcc, 2.0, (200, 128))
# 	video_writer = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 1, (775, 512))
	video_writer = cv2.VideoWriter('output_video_two_pri.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 5, (2*mysim.radar.min_doppler_samples_for_display_one_sided,mysim.radar.min_range_samples_for_display_bb))

	while True:
		# Convert data to a BGR image
		#image = cv2.merge([data, data, data])
		zoa,aoa,d = mysim.target.get_target_entity_geo(mysim.radar.transmitter)
		steered_az = aoa
		steered_z = zoa
		data = np.zeros([mysim.radar.min_range_samples_for_display_bb,2*mysim.radar.min_doppler_samples_for_display_one_sided]) + 0j
		for pri_idx in np.arange(mysim.radar.num_unique_wfs_per_mode):
			x,B = mysim.process_mode_cpi(steered_az,steered_z)
			data += x
			#data += x 
		#mysim.target.update_state(.1)
		h,w = data.shape
		size = (w,h)
		#mysim.target.update_state(.001389) #actual time elapsed for the pri scheme i used
		
		data = np.log10(np.abs(data))
		data = data/np.max(data) * 255.0
		data = data.astype(np.uint8)
		image = cv2.merge([data,data,data])
# 		cv2.line(image, (0, int(h/2)+1), (w, int(h/2)+1), (255, 255, 255), thickness=1)
# 		cv2.line(image, (0, int(h/2)-1), (w, int(h/2)-1), (255, 255, 255), thickness=1)
# 		cv2.line(image, (int(w/2)-1, 0), (int(w/2)-1, h), (255, 255, 255), thickness=1)
# 		cv2.line(image, (int(w/2)+1, 0), (int(w/2)+1, h), (255, 255, 255), thickness=1)
		cv2.line(image, (int(w/2), 0), (int(w/2), h), (0, 0, 0), thickness=1)
		# Display the image
		cv2.imshow(window_name, image)

		# Write the frame to the video file
		video_writer.write(image)

		# Exit when 'q' is pressed
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# Release video writer and close windows
	video_writer.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
	


