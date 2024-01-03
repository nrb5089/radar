import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import butter, cheby1, lfilter,firwin
from copy import deepcopy as dcp
import cv2
import core
from core import BasicReceiver, BasicTransmitter, PlanarAESA
from sim import Simulation
from util import ffts, affts, init_figs

init_figs()

def sincantennapattern_test():
	azimuth_beam_width = 15 * np.pi/180
	elevation_beam_width = 25 * np.pi/180
	peak_antenna_gain_db = 0
	first_side_lobe_down_az_db = 10
	first_side_lobe_down_el_db = 8
	second_side_lobe_down_az_db = 15
	second_side_lobe_down_el_db = 12
	back_lobe_down_db = 20
	mypattern = core.SincAntennaPattern(azimuth_beam_width,
									elevation_beam_width,
									peak_antenna_gain_db,
									first_side_lobe_down_az_db,
									first_side_lobe_down_el_db,
									second_side_lobe_down_az_db,
									second_side_lobe_down_el_db,
									back_lobe_down_db)
	thetas_az = np.linspace(0,np.pi,250)
	thetas_el = np.linspace(0,np.pi,250)
	
	#Slice
	fig,axes = plt.subplots(1,1)
	axes.plot(180/np.pi * thetas_az,10*np.log10(np.abs(mypattern.azimuth_slice(thetas_az))))
	# axes.vlines(slnum*azimuth_beam_width/2*180/np.pi,0,1)
	
	#2D Pattern Demo
	# full_pattern = []
	# for theta in thetas_el:
		# full_pattern.append(10*np.log10(np.abs(mypattern.azimuth_slice(thetas_az,elevation_angle=theta))))
	# full_pattern = np.vstack(full_pattern)
	# fig,axes = plt.subplots()
	# axes.imshow(full_pattern)

# def visualize_doppler_waveforms():
	
	
def lfm_demo():
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

		
def test_array():
	frequency = 9.6e9
	lam = 3e8/frequency
	myaesa = PlanarAESA(100, lam/2)
	aoa = np.pi/4
	zoa = .5
	x = myaesa.array_response(aoa,zoa,frequency)
	a = myaesa.array_response
	mag_response = []
	for phiz in np.linspace(-np.pi/4,3*np.pi/4,100):
		mag_z = []
		for phia in np.linspace(-np.pi/2,np.pi/2,100)[-1::-1]:
			#mag_z.append(np.abs(np.conj(a(phia,phiz,frequency)) @ x ))
			mag_z.append(20*np.log10(np.abs(np.conj(a(phia,phiz,frequency)) @ x )))
		mag_response.append(np.vstack(mag_z))
	#mag_response = np.vstack(mag_z)
	fig,ax = plt.subplots()
	ax.imshow(mag_response)

def visualize_array_geometry():
	fig,ax = plt.subplots(subplot_kw={'projection': '3d'})
# 	for x,z in zip(rx,rz):
# 		for y in ry:
# 			ax.plot(x,y,z,'kx',markersize = 10)
	myaesa = PlanarAESA(16, .1)
	rvec = myaesa.rvec
	for r in rvec:
		ax.plot(r[0],r[1],r[2],'kx',markersize = 10)
	ax.set_ylabel('y')
	ax.set_xlabel('x')
	ax.set_zlabel('z')

def test_cfar2D_window_construction():
	num_reference_cells_x_one_sided = 13
	num_reference_cells_y_one_sided = 10
	num_guard_cells_x_one_sided = 3
	num_guard_cells_y_one_sided = 5
	probability_false_alarm = 1e-6
	mycfar = core.CA_CFAR2D(num_reference_cells_x_one_sided, num_reference_cells_y_one_sided, num_guard_cells_x_one_sided, num_guard_cells_y_one_sided, probability_false_alarm)
	
	fig,ax = plt.subplots()
	ax.imshow(mycfar.cfar_window)	

def test_sinc():
	theta = np.linspace(0,np.pi,1000)
	fig,ax = plt.subplots(1,1)
	ax.plot(theta,np.abs(np.sinc(theta)))
if __name__ == '__main__':
	
	#test_path()
	#test_array()
	#demo_doppler_maps()
	#tracking_sim()
	#myradar = Receiver()
	#x = myradar.pd_wf_object.wf
	#x = myradar.process_signal(x)
	#x = myradar.process_probe_signal(x,myradar.pd_wf_object)
	#x = myradar.test()
	#test_cfar2D_window_construction()
	sincantennapattern_test()
# 	test_sinc()
	plt.show()
	


