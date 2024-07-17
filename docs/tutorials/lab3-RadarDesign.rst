=======================
Radar Search and Track
=======================

The search function of a radar system refers to its capability to scan a volume of space to detect and locate targets. This function is crucial in various applications, including air traffic control, weather monitoring, military surveillance, and maritime navigation. The implementation of the search function can vary depending on the specific type of radar and its intended use. Here’s a general description of how a search function typically works in a radar system:

Scanning the Area or Volume:
=============================

   - **Mechanical Scanning:** Traditional radar systems use a rotating antenna to sweep a radar beam across the sky or sea surface. The rotation speed and the beamwidth determine how quickly an area is scanned.
   - **Electronic Scanning (Phased Array):** More advanced systems use electronically steered phased array antennas. These systems can steer the radar beam rapidly in different directions without moving the antenna, allowing for quicker and more flexible scanning.

In radar systems, especially sophisticated ones like phased array radars, the **scheduling hierarchy** is crucial for effective surveillance and tracking. This hierarchy typically involves organizing scan positions, dwells, and multiple Pulse Repetition Intervals (PRIs) within each dwell. Here's an overview of how this hierarchy is structured:

**Scan Positions**
   - **Definition**: A scan position refers to a specific orientation or angle of the radar beam. In mechanically steered radars, this would be a physical position of the antenna. In phased array radars, it refers to the beam's electronic steering to a particular azimuth and elevation.
   - **Purpose**: By changing scan positions, the radar covers different areas of the surveillance volume.
   - **Scheduling**: The radar system schedules scan positions to ensure complete coverage of the search area. This can be done in a predetermined pattern or adaptively based on the situation (e.g., focusing on areas of interest).

**Dwells**
   - **Definition**: A dwell is a period during which the radar beam is focused on a specific scan position. During a dwell, the radar transmits and receives multiple pulses to gather data from that position.
   - **Purpose**: Dwelling allows the radar to collect enough data to determine target information at that scan position, including range, velocity (through Doppler processing), and sometimes angular information.
   - **Scheduling**: The duration and frequency of dwells are scheduled based on operational requirements. Longer dwells can provide more data (improving detection and resolution) but reduce the radar's ability to quickly scan other areas.

**Multiple PRIs per Dwell Position**
   - **Definition**: Within each dwell, the radar may use multiple PRIs. The PRI is the time interval between consecutive radar pulses. Using multiple PRIs helps in **resolving range and velocity ambiguities**.
   - **Purpose**: By varying the PRI, the radar can distinguish between targets that would otherwise appear in the same range or velocity bins due to the folding effect in range or Doppler processing.
   - **Scheduling**: The selection and scheduling of PRIs within a dwell are critical. The pattern of PRIs can be staggered or switched between different values to optimize ambiguity resolution. This scheduling is often based on algorithms designed to maximize target detection and resolution while managing ambiguities.

**Combined Hierarchy**
- In operation, the radar system schedules a series of scan positions, covering the required search area.
- At each scan position, the radar dwells for a certain time, transmitting and receiving multiple pulses.
- Within each dwell, the radar cycles through a sequence of PRIs, adapting as necessary to resolve ambiguities and optimize target detection.

This hierarchical scheduling allows radar systems to balance the competing needs of area coverage, target detection, and target tracking. Advanced radar systems, especially those with electronic beam steering, can dynamically adjust this hierarchy based on real-time data and mission priorities. For example, a radar might momentarily focus on an area of interest with longer dwells and specific PRI patterns before resuming its broader search pattern.

**Specialized Features:**
- **Search Patterns:** Some radars can perform specialized search patterns, like sector scans (scanning a specific sector more intensely) or random search patterns (to avoid detection or jamming in military applications).
- **Resolution and Accuracy:** Higher resolution radars can distinguish between closely spaced targets and provide more accurate position information.
- **Integration with Other Systems:** Radars are often integrated with other sensor systems and databases for enhanced target identification and situational awareness.

The specific implementation of a radar's search function will depend on its intended application and technological capabilities. Advanced systems can perform complex search patterns and integrate data from multiple sources for a comprehensive understanding of the scanned environment.

**Bottom Line: At the end of the day, any hirarchical scheduler is just a nested state machine**

In this section we decide to introduce the problems within the sections themselves


Please use the following parameters for the following design problems:

.. code-block:: python

    radar_params = {}

    #Transmitter Parameters
    radar_params['x_loc_m_tx'] = 0.0
    radar_params['y_loc_m_tx'] = 0.0
    radar_params['z_loc_m_tx'] = 3.0
    radar_params['x_vel_mps_tx'] = 0.0
    radar_params['y_vel_mps_tx'] = 0.0
    radar_params['z_vel_mps_tx'] = 0.0
    radar_params['x_acc_mps2_tx'] = 0.0
    radar_params['y_acc_mps2_tx'] = 0.0
    radar_params['z_acc_mps2_tx'] = 0.0
    radar_params['rf_sampling_frequency_hz'] = 500e6
    radar_params['if_sampling_frequency_hz'] = 100e6
    radar_params['bb_sampling_frequency_hz'] = 25e6
    radar_params['rf_center_frequency_hz'] = 115e6
    radar_params['rf_bandwidth_hz'] = 20e6
    radar_params['transmit_power_w'] = 100 #per element
    radar_params['internal_loss_db_tx'] = 2

    #Receiver Parameters
    radar_params['x_loc_m_rx'] = 0.0 
    radar_params['y_loc_m_rx'] = 0.0
    radar_params['z_loc_m_rx'] = 3.0
    radar_params['x_vel_mps_rx'] = 0.0
    radar_params['y_vel_mps_rx'] = 0.0
    radar_params['z_vel_mps_rx'] = 0.0
    radar_params['x_acc_mps2_rx'] = 0.0
    radar_params['y_acc_mps2_rx'] = 0.0
    radar_params['z_acc_mps2_rx'] = 0.0
    radar_params['rf_sampling_frequency_hz'] = 500e6
    radar_params['if_sampling_frequency_hz'] = 100e6
    radar_params['bb_sampling_frequency_hz'] = 25e6
    radar_params['rf_center_frequency_hz'] = 115e6
    radar_params['rf_bandwidth_hz'] = 20e6
    radar_params['internal_loss_db_rx'] = 2

    #Detection
    radar_params['num_reference_cells_range_one_sided'] = 20
    radar_params['num_guard_cells_range_one_sided'] = 7
    radar_params['num_reference_cells_doppler_one_sided'] = 10
    radar_params['num_guard_cells_doppler_one_sided'] = 3
    radar_params['probability_false_alarm'] = 1e-3
    radar_params['probability_false_alarm_2D'] = 1e-2
    radar_params['detector_type'] = 'square'

    #Scatterer/Target
    target_params = {}
    target_params['x_loc_m'] = 31000.0 #100 nmi
    target_params['y_loc_m'] = 0
    target_params['z_loc_m'] = 10668 #35kft
    target_params['x_vel_mps'] = -250 #550 knots Remember this is relative to the radar
    target_params['y_vel_mps'] = 0
    target_params['z_vel_mps'] = 0.0
    target_params['x_acc_mps2'] = .1
    target_params['y_acc_mps2'] = .001
    target_params['z_acc_mps2'] = .001
    target_params['radar_cross_section_dbsm'] =25
    

Design Example: Basic 360 Azimuth Scan
========================================

![Alt text](../figs/ppi_scope.png?raw=true)

Design a 360 degree continuous scan (approximation). This is the process that genrates the data to eventually display on a Planned Position Indicator (PPI) as shown above (image source from https://en.wikipedia.org/wiki/File:Ppi_scope.png).  Within your simulation ```init```, initialize a sinc antenna pattern, which is a generic pattern for several directional antennas (i.e., Yagi, horn).

The parameter dicts passed to your ```init``` looks like:

.. code-block:: python

    antenna_params = {}
    antenna_params['azimuth_beam_width'] = 15 * np.pi/180
    antenna_params['elevation_beam_width'] = 25 * np.pi/180
    antenna_params['peak_antenna_gain_db'] = 0
    antenna_params['first_side_lobe_down_az_db'] = 10
    antenna_params['first_side_lobe_down_el_db'] = 8
    antenna_params['second_side_lobe_down_az_db'] = 15
    antenna_params['second_side_lobe_down_el_db'] = 12
    antenna_params['back_lobe_down_db'] = 20


    radar_params['wf_list'] = [{'index': 0, 'type': 'single', 'pw': 100e-6, 'pri': 1500e-6, 'lfm_excursion' : 2e6, 'pris_per_cpi': 1},
                            {'index': 1, 'type': 'single', 'pw': 100e-6, 'pri': 1550e-6, 'lfm_excursion' : 2e6, 'pris_per_cpi': 1},
                            {'index': 2, 'type': 'single', 'pw': 100e-6, 'pri': 1100e-6, 'lfm_excursion' : 2e6, 'pris_per_cpi': 1}]
                            
    radar_params['wf_sequences'] = [{'index': 0, 'type' : 'single_pulse_stagger', 'sequence' : [0]},
                          {'index': 1, 'type' : 'single_pulse_stagger', 'sequence' : [0,1,2,0,1,2,1]}]
                          



the required imports followed by the class and within ```init```,

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    from copy import deepcopy as dcp
    from core import MonostaticRadar, SincAntennaPattern

    class Simulation:
        '''
        Top level simulation class for a 1v1 target vs track radar
        '''
        def __init__(self, sim_params, target_params, radar_params, antenna_params, demo = False):
            
            self.sim_params = sim_params
            self.target_params = target_params
            self.radar_params = radar_params
            self.antenna_params = antenna_params
            
            self.target = DWNATarget(target_params)
            self.radar = MonostaticRadar(radar_params)
            self.antenna_pattern = SincAntennaPattern(antenna_params)
                                        
            self.process_rf = sim_params['process_rf'] 
            self.process_lam = 3e8/self.process_rf


Note the line 

.. code-block:: python

    self.antenna_pattern = SincAntennaPattern(antenna_params)


Feel free to look at the object within ```core``` where the ```SincAntennaPattern``` object resides.  

We are approximating a continuous scan, meaning that in reality the antenna is moving during the PRI.  However, the position change of the antenna during round trip time (PRI) on the order of 1000s of microseconds is negligible for the fidelity we usually work in.  If you feel like splitting hairs on this, feel free to waste your life and time.  A template for a single scan is shown below, we wish the scan to be an integer multiple of CPIs, henc

.. code-block:: python

    def perform_360_scan(self,full_rotation_time_s, ...):

        length_cpi_s = np.sum(self.radar.pris_in_mode_sequence)
        num_cpis_per_rotation = int(np.round(full_rotation_time_s/length_cpi_s))
        steered_azs = np.linspace(0,2*np.pi,num_cpis_per_rotation)
        actual_rotation_time_s = num_cpis_per_rotation * length_cpi_s
        
        ...
        
        for steered_az in steered_azs:
            x, wf_object = self.radar.waveform_scheduler_cpi()
            
            ...
            
            #Tx/Rx Antenna Gain
            x = self.antenna_pattern.gain_val(steered_az-aoa,steered_z-zoa)**2 * x
            
            ...
            
            #Target motion
            self.target.update_state(wf_object.cpi_duration_s)


For all intents and purposes, you can generate a lot of data depending on the granularity of your scan.  I do not necessarily suggest trying to build a PPI... 

Let's take a closer look at what we are doing in the main loop, at each ```steered_az```, we produce a ```wf_object``` that comprises of the waveform for a particular PRI.  We defined waveforms in the nested dictionary ```radar_params['wf_list']```, and the sequences in which these waveforms are traversed in ```radar_params['wf_sequences']```.  Upon instantiating ```MonostaticRadar```, an attribute ```self.wf_bank``` is created that is a **list of waveform objects** (```wf_object```).  Upon calling ```self.radar.waveform_scheduler_cpi```, the next ```wf_object```, as prescribed in ```radar_params['wf_sequences']```, is produced.  This is effectively how the radar scheduler has been implemented and can be manipulated using these tools.   We will cover examples of other waveform modes other than ```single_pulse_stagger``` to come.



Design Example: Dwell and Switch
====================================

![Alt text](../figs/dwell_and_switch.png?raw=true)

In reality, were more concerned with radars that perform electronic scanning over a sector.  The radar may also task schedule share between tracking assignments, which we will discuss later.  The above illustration shows such a scenario of multiple tasking (image source https://www.researchgate.net/figure/RT1-Mode-Volume-Search-Mode-with-Rotating-Antenna_fig1_224127198).


### Problem: Multi-PRI Processing


# Radar Tracking

## Track Association 

## Filtering

