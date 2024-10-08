import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import butter, cheby1, lfilter, firwin, convolve2d
from copy import deepcopy as dcp
#import cv2
from .util import ffts,affts,log2lin


__version__ = "0.1.0"


class PlanarAESA:
    '''
    Calculates the gain for a particular set of azimuth and elevation angles
    
    Not intended to be a true time delay model
    
    By default produces a square grid array
    '''
    def __init__(self,one_sided_elements, array_element_spacing):
        self.num_elements = int(one_sided_elements**2)
        self.d = array_element_spacing
        self.rvec = self.build_rvec(one_sided_elements,array_element_spacing)

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

class CustomAESA:
    '''
    Calculates the gain for a particular set of azimuth and elevation angles
    
    Not intended to be a true time delay model
    
    Accepts 'array_elements_file' where each line is 3 elements representing the x y z coordinates
    of an array element.
    
    '''
    def __init__(self,path_array_elements_file):
        self.rvec = np.loadtxt(path_array_elements_file)
        print(self.rvec)
        
    def array_response(self,azimth_angle,zenith_angle,frequency):
        lam = 3e8/frequency
        kvec = 2*np.pi/lam * np.array([np.sin(zenith_angle)*np.cos(azimth_angle),np.sin(zenith_angle)*np.sin(azimth_angle),np.cos(zenith_angle)])
        return np.exp(1j*self.rvec @ kvec)
        
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
    def __init__(self,params):
        self.params = params
                #Spatial Parameters
                # x_loc_m = 0.0, 
                # y_loc_m = 0.0,
                # z_loc_m = 3.0, 
                # x_vel_mps = 0.0,
                # y_vel_mps = 0.0,
                # z_vel_mps = 0.0,
                # x_acc_mps2 = 0.0,
                # y_acc_mps2 = 0.0,
                # z_acc_mps2 = 0.0,
                
                # #Transmitter and Sampling Parameters
                # rf_sampling_frequency_hz = 32e9,
                # if_sampling_frequency_hz = 4e9,
                # bb_sampling_frequency_hz = 25e6,
                # rf_center_frequency_hz = 13.2e9,
                # rf_bandwidth_hz = 1e9,
                # transmit_power_w = 100, #per element
                # internal_loss_db = 2):
                
        self.state = np.array([params['x_loc_m'],params['y_loc_m'],params['z_loc_m'],params['x_vel_mps'],params['y_vel_mps'],params['z_vel_mps']]) 
        self.Fs_rf = params['rf_sampling_frequency_hz']
        self.Fs_if = params['if_sampling_frequency_hz']
        self.Fs_bb = params['bb_sampling_frequency_hz']
        self.fc_rf = params['rf_center_frequency_hz']
        self.fc_if = np.mod(params['rf_center_frequency_hz'],params['if_sampling_frequency_hz'])
        self.rf_bw = params['rf_bandwidth_hz']
        self.Ptx = params['transmit_power_w']
        self.Ltx = 10**(params['internal_loss_db']/10)

    def transmit_waveform(self,wf_object):
        return np.sqrt(self.Ptx) * wf_object.wf()
        
class BasicReceiver:
    '''
    Transmits a sequence of lfm pulses as a pulse-Doppler burst and executes basic MTI and FFT processing
    '''
    #TODO: Break it up into an "ACTUAL" RF center frequency and a "CALCULATIONS" RF Center frequency
    def __init__(self,params):
        self.params = params
        
    
                # #Spatial Parameters
                # x_loc_m = 0.0, 
                # y_loc_m = 0.0,
                # z_loc_m = 3.0, 
                # x_vel_mps = 0.0,
                # y_vel_mps = 0.0,
                # z_vel_mps = 0.0,
                # x_acc_mps2 = 0.0,
                # y_acc_mps2 = 0.0,
                # z_acc_mps2 = 0.0,  
                
                # #Receiver and Sampling Parameters
                # rf_sampling_frequency_hz = 32e9,
                # if_sampling_frequency_hz = 4e9,
                # bb_sampling_frequency_hz = 25e6,
                # rf_center_frequency_hz = 13.2e9,
                # rf_bandwidth_hz = 1e9,
                # receiver_noise_figure_db = 5,
                # internal_loss_db = 2,
                
                # # CFAR Parameters
                # num_reference_cells_range_one_sided = 10,
                # num_guard_cells_range_one_sided = 4,
                # probability_false_alarm = 1e-6,
                # num_reference_cells_doppler_one_sided = 10,
                # num_guard_cells_doppler_one_sided = 3,
                # probability_false_alarm_2D = 1e-1,
                # detector_type = 'square'):
        
        self.state = np.array([params['x_loc_m'],params['y_loc_m'],params['z_loc_m'],params['x_vel_mps'],params['y_vel_mps'],params['z_vel_mps']]) 
        self.Fs_rf = params['rf_sampling_frequency_hz']
        self.Fs_if = params['if_sampling_frequency_hz']
        self.Fs_bb = params['bb_sampling_frequency_hz']
        self.fc_rf = params['rf_center_frequency_hz']
        self.fc_if = np.mod(params['rf_center_frequency_hz'],params['if_sampling_frequency_hz'])
        self.rf_bw = params['rf_bandwidth_hz']
        self.det_type = params['detector_type']
        
        self.rf_lam = 3e8/self.fc_rf
        self.rf2if_ds = int(self.Fs_rf/self.Fs_if)
        self.if2bb_ds = int(self.Fs_if/self.Fs_bb)
        
        self.NF_lin = 10**(params['receiver_noise_figure_db']/10)
        self.Lrx = 10**(params['internal_loss_db']/10)
        
        #TODO Add Power limiting, like a tanh
        
        self.sigma_n = np.sqrt(1.38e-23 * 290 * params['rf_bandwidth_hz'] * self.NF_lin)
        
        #Analog Front End Filters
        #assert self.fc_if + self.rf_bw/2 < self.Fs_rf/2, f'Top of RF signal bandwidth {(self.fc_rf + self.rf_bw/2 )/2/1e6} MHz equals or exceeds sampling Nyquist limit {self.Fs_rf/2/1e6} MHz!'
        self.rf2if_filter = ButterFilter(N = 2,Wn = [self.fc_rf - self.rf_bw/2,self.fc_rf + self.rf_bw/2],fs = self.Fs_rf,btype = 'bandpass')
        self.adc_filter = Cheby1Filter(N = 5,rp = 2, Wn = self.Fs_if/2 -10e6,fs = self.Fs_if, btype = 'low')
        
        #Doppler Processing
        self.mti = MTIBasic()
        
        # #Detection
        self.cfar = CA_CFAR1D(params['num_reference_cells_range_one_sided'], params['num_guard_cells_range_one_sided'],params['probability_false_alarm'])
        self.cfar2D = CA_CFAR2D(params['num_reference_cells_doppler_one_sided'], 
                                params['num_reference_cells_range_one_sided'],
                                params['num_guard_cells_doppler_one_sided'],
                                params['num_guard_cells_range_one_sided'],
                                params['probability_false_alarm_2D'])
        
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
        x2D = np.fft.fftshift(np.fft.fft(x2D),axes = 1)
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
        

class Waveform:
    """
    A class to represent a radar waveform.
    
    

    Attributes
    ----------
    index: int
        Unique index given to distinguish among other Waveform objects used.
    
    type : str
        Indicates the waveform type being a single pulse or a burst of pulses. Valid options are 'single' and 'burst'.
        
    pw : float
        Pulse Width (PW) in seconds.
    
    pri : float
        Pulse Repetition Interval (PRI) in seconds.
        
    modulation : str
        Specifies the type of modulation on pulse used.  Valid options are 'bpsk', 'lfm', or 'none'.
    
    lfm_ex : float
        Specifies the linear frequency modulation excursion (i.e., 1 MHz is a 1 MHz chirp or lfm) in Hertz. Not used if modulation is not set to 'lfm'.
        
    bpsk_seq : numpy.ndarray
        Specifies the binary phase shift keyed sequence consisting of '-1' and '1' as a numpy array, i.e., a 13-bit Barker code would be np.array([1,1,1,1,1,-1,-1,-1,1,-1,1,-1,1]). dtype should be float32, float64, or int. Not used if modulation is not set to 'bpsk'.
        
    chip_width_s : float
        Specifies the length of a bpsk chip in seconds.  Not used if modulation is not set to 'bpsk'.
        
    pris_per_cpi : int
        Specifies the number of PRIs in a Coherent Processing Interval (CPI).  For waveform type 'single', this defaults to 1.

    Fs_rf : float
        Sampling frequency at transmit/receive (RF) frequency in Hertz.
    
    Fs_if : float
        Sampling frequency at relevant intermediate freqency (IF) in Hertz.
    
    Fs_bb : float
        Sampling frequency at baseband (BB) in Hertz.
    
    fc_rf : float
        RF center frequency in Hertz.
        
    fc_if : float
        IF center frequency in Hertz.
    
    samples_per_chip_<freq> : int 
        Computed by: int(np.ceil(self.pw/self.bpsk_length * self.Fs_<freq>)). <freq> can be 'bb', 'if', or 'rf'.
            
    samples_per_pw_<freq> : int
        Computed by: int(self.pw * self.Fs_<freq>). <freq> can be 'bb', 'if', or 'rf'.
        
    samples_per_pri_<freq> : int
        Computed by: int(self.pri * self.Fs_<freq>). <freq> can be 'bb', 'if', or 'rf'.
        
    samples_per_cpi_<freq> : int
        Computed by: int(self.pris_per_cpi * self.samples_per_pri_<freq>). <freq> can be 'bb', 'if', or 'rf'.
    
    cpi_duration_s : float
        Duration of CPI in seconds
        
    samples_per_range_window_<freq>: int
        Computed by: samples_per_pri_<freq> - int(2 * self.samples_per_pw_<freq>).  <freq> can be 'bb', 'if', or 'rf'.
            
    range_per_cpi_m : float
        maximum echo range within a CPI in meters, determined by c * cpi_duration_s/2
        
    range_unambiguous : float
        maximum unambiguous range in meters, determined by cPRI/2
    
    bb_filter : FIR
        An instance of FIR representing the filter used for the low pass filtering of the baseband waveform.
    
    mf_wf_bb : numpy.ndarray
        Baseband waveform to match pulse used in matched filter.  Applied using method 'apply_matched_filter'. dtype should be complex64 or complex128.
    
    range_resolution : float
        Range resolution in meters. Computed by: 3e8/2/self.bandwidth.
    
    rcv_window_mask : numpy.ndarray
        Array of 1s and 0s that indicate the the listening on and off intervals, respectively, for the waveform to be received.  
    
    Methods
    -------
    wf():
        Generates samples at the RF center frequency (fc_rf) at RF sampling frequency (Fs_rf) for one coherent processing interval.  For type 'single', this is just a single pri, for type 'burst', this is a train of pulses.
        
    apply_matched_filter(x):
        Correlates a single matched pulse to the incoming signal x, as specified by waveform attributes. Filtering performed as Baseband (BB) sampling rate 
        
    apply_bb_filter(x):
        Applies Baseband (BB) low-pass filter limited to waveform bandwidth.  For LFM, this is the lfm_ex, for BPSK, this is 1/chip_width_s, and for unmodulated, this is just 1/pw
        
        
    Notes
    -------
    When we refer to the RF center frequency or RF sampling rate, we intend the frequency at which the waveform is transmitted, i.e., the antenna(s) and front end tuned such that it utilizes that frequency.
    
    The range window is the time the radar spends listening for pulses echoed.  For burst waveforms, there is an inherently restricted range due to the need to listen for the full pulse length and then transmit the second pulse, resulting in what are called its 'blind zones'.  These are accounted for by applying the self.rcv_window_mask to an incoming signal x.
    
    Examples
    ---------
    wf_params: 
    lfm_single_wf_params = {'index': 2, 'type': 'single', 'pw': 100e-6, 'pri': 1100e-6, 'modulation' : 'lfm', 'lfm_excursion' : 2e6, 'bpsk_seq' : [], 'bpsk_chipw' : 0.,'pris_per_cpi': 1}
    bpsk_burst_wf_params = {'index' : 5,'type': 'burst', 'pw': .75e-06, 'pri': 4e-6, 'modulation' : 'bpsk', 'lfm_excursion' : 0., 'bpsk_seq' : [1,1,1,1,1,-1,-1,1,1,-1,1,-1,1], 'bpsk_chipw' : .04e-6,'pris_per_cpi': 200}
    
    Example transmit/receive chain for Waveform instance mywf:
    x = mywf.wf() 						#Transmited waveform
    x = applyEnvironment(x)  			#function specific to user scenario
    x = mywf.apply_rcv_mask(x) 			#Account for blind range zones
    x = applyRFFrontend(x) 				#function specific to user scenario
    x = convert2baseband(x) 			#function specific to user scenario
    x = mywf.apply_bb_filter(x)			#Filter at baseband
    x = mywf.apply_matched_filter(x)	#Matched filter for pulse
    """

    def __init__(self, radar_params, wf_params):
        """
        Constructs all necessary attributes.
        
        Parameters
        ----------
            radar_params : dict
                A dictionary containing radar signal processing parameters.  Expected key-value pairs:
                    'rf_sampling_frequency_hz' (float): Sampling frequency at transmit/receive (RF) frequency in Hertz.
                    'if_sampling_frequency_hz' (float): Sampling frequency at relevant intermediate freqency (IF) in Hertz.
                    'bb_sampling_frequency_hz' (float): Sampling frequency at baseband (BB) in Hertz.
                    'fc_rf' (float): RF center frequency in Hertz.
                    
            wf_params : dict
                A dictionary containing waveform parameter information.  Expected key-value pairs:
                    'index' (int): Unique index given to distinguish among other Waveform objects used.
                    'type' (str): Indicates the waveform type being a single pulse or a burst of pulses. Valid options are 'single' and 'burst'.
                    'pw' (float): Pulse Width (PW) in seconds.
                    'pri' (float): Pulse Repetition Interval (PRI) in seconds.
                    'modulation' (str): Specifies the type of modulation on pulse used.  Valid options are 'bpsk', 'lfm', or 'none'.
                    'lfm_ex' (float): Specifies the linear frequency modulation excursion (i.e., 1 MHz is a 1 MHz chirp or lfm) in Hertz. Not used if modulation is not set to 'lfm'.
                    'bpsk_seq' (numpy.ndarray): Specifies the binary phase shift keyed sequence consisting of '-1' and '1' as a numpy array, i.e., a 13-bit Barker code would be np.array([1,1,1,1,1,-1,-1,-1,1,-1,1,-1,1]). dtype should be float32, float64, or int. Not used if modulation is not set to 'bpsk'.
                    'chip_width_s' (float): Specifies the length of a bpsk chip in seconds.  Not used if modulation is not set to 'bpsk'.
                    'pris_per_cpi' (int): Specifies the number of PRIs in a Coherent Processing Interval (CPI).  For waveform type 'single', this defaults to 1.
        """
                
        self.index = wf_params['index'] #A radar may have multiple waveforms that it assigns unique indices to each
        self.type = wf_params['type'] #burst or single
        self.pw = wf_params['pw'] #pulse width in seconds, ignored if modulation type bpsk, pulse width is equal to chip_width_s * len(bpsk_seq)
        self.pri = wf_params['pri'] #pulse repetition interval in seconds
        self.modulation = wf_params['modulation'] #modulation type: lfm, bpsk, or none
        self.lfm_ex = wf_params['lfm_excursion'] #linear frequency modulation (lfm) excursion in Hz, ignored if not modulation type lfm
        self.bpsk_seq = wf_params['bpsk_seq'] #Array or 1s and -1s of the bpsk sequence, ignored if not modulation type bpsk 
        self.chip_width_s = wf_params['bpsk_chipw'] #Chip width for bpsk symbols, used to determine pulse width for bpsk waveform types
        self.pris_per_cpi = wf_params['pris_per_cpi'] #number of pris in coherent processing interval (cpi), for Doppler (burst) pulses, this is the total number of pulses in the burst
        
        #parameters translated from the radar employing the waveform
        self.Fs_rf = radar_params['rf_sampling_frequency_hz']
        self.Fs_if = radar_params['if_sampling_frequency_hz']
        self.Fs_bb = radar_params['bb_sampling_frequency_hz']
        self.fc_rf = radar_params['rf_center_frequency_hz']
        self.fc_if = np.mod(self.fc_rf,self.Fs_if)
        
        if self.modulation == 'bpsk':
            self.bpsk_length = len(self.bpsk_seq)
            self.pw = self.bpsk_length * self.chip_width_s
            self.samples_per_chip_bb = int(np.ceil(self.pw/self.bpsk_length * self.Fs_bb))
            self.samples_per_chip_if = int(self.samples_per_chip_bb * self.Fs_if/self.Fs_bb)
            self.samples_per_chip_rf = int(self.samples_per_chip_if * self.Fs_rf/self.Fs_if)
            
        self.samples_per_pw_rf = int(self.pw * self.Fs_rf)
        self.samples_per_pw_if = int(self.pw * self.Fs_if)
        self.samples_per_pw_bb = int(self.pw * self.Fs_bb)
        
        self.samples_per_pri_rf = int(self.pri * self.Fs_rf)
        self.samples_per_pri_if = int(self.pri * self.Fs_if)
        self.samples_per_pri_bb = int(self.pri * self.Fs_bb)
        
        self.samples_per_cpi_rf = int(self.pris_per_cpi * self.samples_per_pri_rf)
        self.samples_per_cpi_if = int(self.pris_per_cpi * self.samples_per_pri_if)
        self.samples_per_cpi_bb = int(self.pris_per_cpi * self.samples_per_pri_bb)
        self.cpi_duration_s = self.pris_per_cpi * self.pri
        
        self.samples_per_range_window_rf = self.samples_per_pri_rf - int(2 * self.samples_per_pw_rf)
        self.samples_per_range_window_if = self.samples_per_pri_if - int(2 * self.samples_per_pw_if)
        self.samples_per_range_window_bb = self.samples_per_pri_bb - int(2 * self.samples_per_pw_bb)
        
        
        
        self.range_per_cpi_m = self.samples_per_cpi_rf / self.Fs_rf * 3e8/2
        self.range_unambiguous = 3e8 * self.pri/2
        
        if self.modulation == 'lfm':
            self.bandwidth = dcp(self.lfm_ex)
            self.mf_wf_bb = np.exp(1j * 2 * np.pi/self.Fs_bb * (np.cumsum(np.linspace(-self.bandwidth/2,self.bandwidth/2,self.samples_per_pw_bb))))

        elif self.modulation == 'bpsk':
            self.bpsk_length = len(self.bpsk_seq)
            self.bandwidth = 1/self.chip_width_s
            self.mf_wf_bb = np.repeat(self.bpsk_seq[-1::-1],self.samples_per_chip_bb) + 0j
        elif self.modulation == 'none':
            self.bandwidth = 1/self.pw
        
        self.range_resolution = 3e8/2/self.bandwidth
        
        #Digital Decimation Pre-Filter for MF
        self.bb_filter = FIR(numtaps = 31, cutoff = self.bandwidth/2, fs = self.Fs_bb)
        self.mf_wf_bb = np.ones(self.samples_per_pw_bb) + 0j
        
    
        #Receive window for a PRI starts on the falling edge of the pulse and has a listening time equal to a PRI - 2PW
        self.rcv_window_mask = []
        
        if self.type == 'burst':
            for pri_idx in np.arange(self.pris_per_cpi):
                self.rcv_window_mask.extend(np.zeros(self.samples_per_pw_rf) + 0j)
                self.rcv_window_mask.extend(np.ones(self.samples_per_range_window_rf) + 0j)
                self.rcv_window_mask.extend(np.zeros(int(self.samples_per_pw_rf)) + 0j)
                
        elif self.type == 'single':
            self.rcv_window_mask.extend(np.zeros(self.samples_per_pw_rf) + 0j)
            self.rcv_window_mask.extend(np.ones(self.samples_per_range_window_rf) + 0j)
            self.rcv_window_mask = np.array(self.rcv_window_mask)
        
    def wf(self):
        """
        Generates samples at the RF center frequency (fc_rf) at RF sampling frequency (Fs_rf) for one coherent processing interval.  For type 'single', this is just a single pri, for type 'burst', this is a train of pulses.

        Parameters
        -----------
            None
        
        Returns
        ----------
        wf_out : numpy.ndarray
            Complex digital IQ representation of "single" or "burst" waveform for one CPI.  Length calculated by CPI length and RF sampling frequency. dtype will be complex64 or complex128.
            
        
        Notes
        ----------
        Waveforms set as generators to preserve memory.
        
        """
        
        if self.modulation == 'lfm':
            wf_single_pri = np.concatenate([np.exp(1j * 2 * np.pi/self.Fs_rf * (self.fc_rf *np.arange(self.samples_per_pw_rf) + np.cumsum(np.linspace(-self.bandwidth/2,self.bandwidth/2,self.samples_per_pw_rf)))),np.zeros(self.samples_per_pri_rf-self.samples_per_pw_rf) + 0j])
        
        if self.modulation == 'bpsk':
            wf_single_pri = np.concatenate([np.repeat(self.bpsk_seq,self.samples_per_chip_rf) * np.exp(1j * 2*np.pi/self.Fs_rf * self.fc_rf *np.arange(self.samples_per_pw_rf)),np.zeros(self.samples_per_pri_rf-self.samples_per_pw_rf) + 0j])
        
        if self.modulation == 'none':
            wf_single_pri = np.concatenate([np.exp(1j * 2*np.pi/self.Fs_rf * self.fc_rf *np.arange(self.samples_per_pw_rf)),np.zeros(self.samples_per_pri_rf-self.samples_per_pw_rf) + 0j])
            
        if self.type == 'burst':
            burst_wf = []
            for ii in np.arange(self.pris_per_cpi):
                burst_wf.extend(wf_single_pri)
            wf_out = np.array(burst_wf)
            
        elif self.type == 'single': 
            wf_out = wf_single_pri
            
        return wf_out
        
    def apply_matched_filter(self,x): 
        """
        Correlates a single matched pulse to the incoming signal x, as specified by waveform attributes. Filtering performed as Baseband (BB) sampling rate
        
        Parameters
        -----------
        x : numpy.ndarray
            Complex digital IQ representation of signal. dtype should be complex64 or complex128.
            
        Returns
        -------
        Filtered complex digital IQ representation of signal. dtype will be complex64 or complex128.
        """
        return np.convolve(x,np.conj(self.mf_wf_bb), mode = 'same')
    
    def apply_bb_filter(self,x): 
        """
        Applies Baseband (BB) low-pass filter limited to waveform bandwidth.  For LFM, this is the lfm_ex, for BPSK, this is 1/chip_width_s, and for unmodulated, this is just 1/pw
        
        Parameters
        -----------
        x : numpy.ndarray
            Complex digital IQ representation of signal. dtype should be complex64 or complex128.
            
        Returns
        -------
        Filtered complex digital IQ representation of signal. dtype will be complex64 or complex128.
        """
        return self.bb_filter.filter_signal(x)
    
    def apply_rcv_mask(self,x): 
        """
        Applies self.rcv_window_mask to incoming signal to account for blind range zones.
        
        Parameters
        ------------
        x : numpy.ndarray
            Incoming signal of type complex64 or complex128.
        """
        return x * self.rcv_window_mask
        
    def help(self):
        """Prints the class docstring and the docstrings of its methods."""
        # Print class docstring
        print(f"Class Documentation:\n{self.__class__.__doc__}\n")
        
        # Iterate through the class attributes
        for attr in dir(self):
            # Filter out special and private attributes/methods
            if not attr.startswith("__"):
                # Get the attribute/method object
                attr_obj = getattr(self, attr)
                # If it's callable (a method), print its docstring
                if callable(attr_obj):
                    print(f"Method {attr} Documentation:\n{attr_obj.__doc__}\n")
        
class ButterFilter:
    """
    Class wrapper for scipy.signal.butter class object.
    
    Parameters
    -----------
    N : int
        Number of filter taps.
        
    Wn : float or tuple
        Designates bandlimits of filter, dtype is float if btype is 'low' or 'high'.  For btype 'bandpass', should be a tuple specifying start and stop band in Hertz.
    
    Fs : float
        Sampling frequency in Hertz.
    
    btype : str
        Filter type. Valid options are 'low', 'high', and 'bandpass'.
        
    
    Attributes
    -----------
    N : int
        Number of filter taps.
        
    Wn : float or tuple
        Designates bandlimits of filter, dtype is float if btype is 'low' or 'high'.  For btype 'bandpass', should be a tuple specifying start and stop band in Hertz.
    
    Fs : float
        Sampling frequency in Hertz.
    
    btype : str
        Filter type. Valid options are 'low', 'high', and 'bandpass'.
    
    a : numpy.ndarray
        Numerator coefficients in frequency response. dtype is float32 or float64.
        
    b : numpy.ndarray
        Denominator coefficients in frequency response. dtype os float32 or float64
        
    Methods
    -------
    filter_signal(x)
        Applies scipy.signal.lfilter(self.a,self.b,x) to signal x.  dtype of x should be float32, float 64, complex64, or complex128.
    
    """
    
    def __init__(self,N,Wn,fs,btype):
        self.N = N
        self.Wn = Wn
        self.Fs = fs
        self.btype = btype
        
        self.b,self.a = butter(N = N, Wn = Wn, fs = fs, btype = btype)
    
    def filter_signal(self,x): return lfilter(self.b,self.a,x)

class Cheby1Filter:
    """
    Class wrapper for scipy.signal.cheby1 class object.
    
    Parameters
    -----------
    N : int
        Number of filter taps.
        
    Wn : float or tuple
        Designates bandlimits of filter, dtype is float if btype is 'low' or 'high'.  For btype 'bandpass', should be a tuple specifying start and stop band in Hertz.
    
    Fs : float
        Sampling frequency in Hertz.
        
    rp : float
        Specifies passband ripple in dB.
    
    btype : str
        Filter type. Valid options are 'low', 'high', and 'bandpass'.
    
    Attributes
    -----------
    N : int
        Number of filter taps.
        
    Wn : float or tuple
        Designates bandlimits of filter, dtype is float if btype is 'low' or 'high'.  For btype 'bandpass', should be a tuple specifying start and stop band in Hertz.
    
    Fs : float
        Sampling frequency in Hertz.
        
    rp : float
        Specifies passband ripple in dB.
    
    btype : str
        Filter type. Valid options are 'low', 'high', and 'bandpass'.
    
    a : numpy.ndarray
        Numerator coefficients in frequency response. dtype is float32 or float64.
        
    b : numpy.ndarray
        Denominator coefficients in frequency response. dtype is float32 or float64.
        
    Methods
    --------
    filter_signal(x)
        Applies scipy.signal.lfilter(self.a,self.b,x) to signal x.  dtype of x should be float32, float 64, complex64, or complex128.
    
    """
    def __init__(self,N,rp,Wn,fs,btype):
        self.N = N
        self.rp = rp
        self.Wn = Wn
        self.Fs = fs
        self.btype = btype
        
        self.b,self.a = cheby1(N = N, rp = rp, Wn = Wn, fs = fs, btype = btype)
    
    def filter_signal(self,x): return lfilter(self.b,self.a,x)
    
class FIR:
    """
    Class wrapper for scipy.signal.firwin class object.
    
    Parameters
    -----------
    numtaps : int
        Number of filter taps.
        
    cutoff : float 
        Designates bandlimit of filter, dtype is float with units in Hertz.
    
    fs : float
        Sampling frequency in Hertz.
    
    Attributes
    -----------
    numtaps : int
        Number of filter taps.
        
    cutoff : float 
        Designates bandlimit of filter, dtype is float with units in Hertz.
    
    fs : float
        Sampling frequency in Hertz.
        
    h : numpy.ndarray
        Impulse response filter coefficients. dtype is float32 or float64.
        
    Methods
    --------
    filter_signal(x)
        Applies numpy.convolve(x,self.h) to signal x.  dtype of x should be float32, float 64, complex64, or complex128.
    
    """
    def __init__(self,numtaps, cutoff, fs):
        self.numtaps = numtaps
        self.cutoff = cutoff
        self.Fs = fs
        
        self.h = firwin(numtaps = numtaps, cutoff = cutoff, fs = fs)
    
    def filter_signal(self,x): return np.convolve(x,self.h,mode = 'same')

class CA_CFAR1D:
    """
    One dimensional cell-averaging CFAR, calculates the threshold and builds detection
    vector with input signal.
    
    Parameters
    -----------
    num_reference_cells_one_sided : int
        Number of taps on CFAR window on one side of the cell under test.
        
    num_guard_cells_one_sided : int
        Number of guard cell taps on the CFAR window on one side of the cell under test.
    
    probability_false_alarm : float
        Probability of false alarm that characterizes the CFAR.
    
    Attributes
    -----------
    num_ref : int
        Number of taps on CFAR window on one side of the cell under test.
        
    num_guard : int
        Number of guard cell taps on the CFAR window on one side of the cell under test.
    
    pfa : float
        Probability of false alarm that characterizes the CFAR.
        
    cfar_constant : float
        CFAR constant.
        
    cfar_window : numpy.ndarray
        Array of 1s and 0s that is convolved with incoming signal to induce CA-CFAR operation.
        
    Methods
    --------
    calculate_cfar_thresh(x)
        Applies numpy.convolve(x,self.cfar_window, mode = 'same') to signal numpy.ndarray x.  dtype of x should be float32, float 64, complex64, or complex128.
        
    build_detection_vector(x,T)
        Returns vector consisting of 1s and 0s, where 1 indicates that the signal has broken the detection threshold T, which is an numpy.ndarray.
    
    
    Notes
    ------
    CA-CFAR theory derived from M. A. Richards, Fundamentals of Radar Signal Processing, 2nd ed. New York, NY, USA: McGraw-Hill Education, 2014.

    """
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
    """
    Two dimensional cell-averaging CFAR, calculates the threshold and builds detection
    vector with input signal.
    
    Parameters
    -----------
    num_reference_cells_doppler_one_sided : int
        Number of taps on CFAR window on one side of the cell under test along Doppler dimension.
        
    num_reference_cells_range_one_sided : int
        Number of taps on CFAR window on one side of the cell under test along range dimension.
        
    num_guard_cells_doppler_one_sided : int
        Number of guard cell taps on the CFAR window on one side of the cell under test along Doppler dimension.
        
    num_guard_cells_range_one_sided : int
        Number of guard cell taps on the CFAR window on one side of the cell under test along range dimension.
    
    probability_false_alarm : float
        Probability of false alarm that characterizes the CFAR.
    
    Attributes
    -----------
    num_ref_d : int
        Number of taps on CFAR window on one side of the cell under test along Doppler dimension.
        
    num_ref_r : int
        Number of taps on CFAR window on one side of the cell under test along range dimension.
        
    num_guard_d : int
        Number of guard cell taps on the CFAR window on one side of the cell under test along Doppler dimension.
        
    num_guard_r : int
        Number of guard cell taps on the CFAR window on one side of the cell under test along range dimension.
    
    pfa : float
        Probability of false alarm that characterizes the CFAR.
        
    cfar_constant : float
        CFAR constant.
        
    cfar_window : numpy.ndarray
        Matrix of 1s and 0s that is convolved with incoming signal to induce CA-CFAR operation.
        
    Methods
    --------
    calculate_cfar_thresh(x)
        Applies numpy.convolve(x,self.cfar_window, mode = 'same') to signal numpy.ndarray x.  dtype of x should be float32, float 64, complex64, or complex128.
        
    build_detection_matrix(x,T)
        Returns vector consisting of 1s and 0s, where 1 indicates that the signal has broken the detection threshold T, which is an numpy.ndarray.
    
    
    Notes
    ------
    CA-CFAR theory derived from M. A. Richards, Fundamentals of Radar Signal Processing, 2nd ed. New York, NY, USA: McGraw-Hill Education, 2014.

    """
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
    #init_figs()
    #plt.close('all')
# 	myradar = Receiver(rf_sampling_frequency_hz = 500e6,
# 					if_sampling_frequency_hz = 100e6,
# 					bb_sampling_frequency_hz = 20e6,
# 					rf_center_frequency_hz = 20e6,
# 					rf_bandwidth_hz = 10e6)

    #single_pulse_demo()
    #test_path()
    #test_array()
    #demo_doppler_maps()
    #tracking_sim()
    #myradar = Receiver()
    #x = myradar.pd_wf_object.wf
    #x = myradar.process_signal(x)
    #x = myradar.process_probe_signal(x,myradar.pd_wf_object)
    #x = myradar.test()
    #plt.show()
    myaesa = CustomAESA('test_elements.txt')
    


