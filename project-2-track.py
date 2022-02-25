import os, h5py, numpy
from datetime import datetime
from numpy import zeros, arange, nan, real, imag, conj, pi, sin, arctan, arctan2, angle, exp, sqrt, diff, pad, std
from utilities.gpsl1ca import get_GPS_L1CA_code_sequence, acquire_GPS_L1CA_signal, \
    L1CA_CODE_RATE, L1CA_CODE_LENGTH, L1CA_CARRIER_FREQ
from utilities.file_source import get_file_source_info, SampleLoader
from utilities.gpst import dt2gpst, gpst2dt, gpst_week_seconds
from utilities.hdf5_utils import write_dict_to_hdf5

# other libraries 
import math
from scipy.io import loadmat
import scipy.stats as stats

#%% track_GPS_L1CA_signal_closed function
def track_GPS_L1CA_signal_closed(prn, source_params, acq_sample_index, code_phase_acq, doppler_acq, **kwargs):
    '''
    Given a PRN, acquires and tracks the corresponding GPS L1CA signal.
    
    Inputs:
        `prn` -- the PRN of the GPS satellite to acquire and track
        `source_params` -- dict containing information about the file source
        `acq_sample_index` -- the index of the sample corresponding to the acquired signal parameters
        `code_phase_acq` -- acquired signal code phase in chips
        `doppler_acq` -- acquired signal Doppler frequency in Hz
        
        `N_integration_code_periods` -- number of code periods (default 1) over which to coherently integrate when tracking
        `epl_chip_spacing` -- spacing of the EPL correlators in units of chips (default 0.5)
        `DLL_bandwidth` -- the bandwidth of the DLL (delay-locked loop) loop filter in Hz (default 5)
        `PLL_bandwidth` -- the bandwidth of the PLL (phase-locked loop) filter in Hz (default 20)
    
    Notes:
    
    In order to avoid data bit transitions, our tracking loop will correlate over an integer number of the CA code periods.
    Nominally the code period is 1 ms, which at a sampling rate of 5 MHz comes out to 5000 samples.  However, due to Doppler
    expansion/compression, the actual number of samples will be slightly above or slightly below 5000.  We adjust the time
    step accordingly in the tracking loop, but the nominal time step is sufficient for designing our loop filter.
    
    James' Notes on confusing things(tm): 
        * DLLs are used to detect code phase (via code phase error of current estimate)
        * PLLs are used to detect carrier phase and frequency (via error of current estimates)
    '''   
    
    # 0. Here we set up the tracking loop
    #  Any computations we can do outside the main loop will speed up our code, improve our own lives,
    # and be better for the planet.
    
    # The `sample_loader` object will help us load samples from the binary file stream
    sample_loader = SampleLoader(source_params['samp_rate'], source_params['bit_depth'],
                                 source_params['is_signed'], source_params['is_integer'],
                                 source_params['is_complex'], source_params['i_lsn'])
    
    # Here we define the "tracking block size" as an integer multiple of the L1CA code period.
    # Nominally, the tracking block size will be some multiple of 1 millisecond, and is equal to the
    # coherent integration time as well as the nominal tracking loop update time step.  Due to code
    # expansion / compression, the actual tracking loop update time step will be expanded or
    # compressed by a small amount.  However, this expansion / compression will not affect the design
    # of our estimators or loop filters.
    N_integration_code_periods = kwargs.get('N_integration_code_periods', 1)
    block_length_chips = L1CA_CODE_LENGTH * N_integration_code_periods
    block_duration = block_length_chips / L1CA_CODE_RATE
    integration_time = block_duration

    # Here we define a time array that we use for sampling our reference signal
    N_block_samples = int(integration_time * source_params['samp_rate'])
    block_time = arange(N_block_samples) / source_params['samp_rate']
    N_blocks = source_params['file_length'] // N_block_samples    

    # The EPL correlation delay spacing controls the sensitivity of the DLL to noise vs. multipath.
    # EPL stands for early, prompt, and late (correlators)
    epl_chip_spacing = kwargs.get('epl_chip_spacing', 0.5)

    # Here we define the DLL and PLL loop filter bandwidths, with default values of 5 and 20 Hz, resp.
    DLL_bandwidth = kwargs.get('DLL_bandwidth', 5)
    PLL_bandwidth = kwargs.get('PLL_bandwidth', 20)

    # Here we preallocate our outputs
    output_keys = ['sample_index', 'early', 'prompt', 'late',
                   'code_phase', 'unfiltered_code_phase', 'filtered_code_phase',
                   'carr_phase', 'unfiltered_carr_phase', 'filtered_carr_phase',
                   'doppler', 'filtered_doppler']
    output_dtypes = [int, complex, complex, complex,
                     float, float, float,
                     float, float, float,
                     float, float]
    outputs = {key: numpy.zeros(N_blocks, dtype=dtype) for key, dtype in zip(output_keys, output_dtypes)}

    # Compute the intermediate frequency
    inter_freq = L1CA_CARRIER_FREQ - source_params['center_freq']
    
    # Get the appropriate PRN code sequence. Comes in [0,1]
    code_seq = get_GPS_L1CA_code_sequence(prn)
    
    # Here we find the next sample corresponding to a start of a data bit transition.  Our
    # tracking will begin at this sample and our initial code phase will be approximately 0.
    code_rate = L1CA_CODE_RATE * (1 + doppler_acq / L1CA_CARRIER_FREQ)
    start_sample = acq_sample_index + \
        int(((-code_phase_acq) % (20 * L1CA_CODE_LENGTH)) * source_params['samp_rate'] / code_rate)
    
    # Set tracking state variables
    code_phase = 0
    code_rate = L1CA_CODE_RATE * (1 + doppler_acq / L1CA_CARRIER_FREQ)
    carr_phase = 0
    doppler = doppler_acq

    sample_index = start_sample
    block_index = 0

    earlys = []
    prompts = []
    lates = []
    # Open the IF sample file
    with open(source_params['filepath'], 'rb') as f:
        
        # Run the tracking loop
        for block_index in range(N_blocks):

            if sample_index + N_block_samples >= source_params['file_length']:
                break
            print('\r {0: >4.1f}'.format(sample_index / source_params['samp_rate']), end='')

            
            # 1. Get the next block of samples
            block = sample_loader.generate_sample_block(f, sample_index, N_block_samples)

            
            # 2. Reference Generation and Correlation
            #  For efficiency, this step is broken down into carrier wipeoff, code wipeoff, and
            # summation.  This process is equivalent to generating complete references and
            # correlating them with our IF samples.

            # 2a. Wipeoff carrier
            #  The reason we do this in a separate step is because the `cos` and `sin` operations
            # are rather expensive (but less expensive than `exp`). So if we can get away with
            # generating our carrier just once, it's worth it.
            phi = 2 * pi * carr_phase + 2 * pi * (inter_freq + doppler) * block_time
            carrier_conj = numpy.cos(phi) - 1j * numpy.sin(phi)     # conjugate of the carrier. Why the complex part? Is this for I/Q channels? 
            block_wo_carrier = block * carrier_conj                 # Mixing down to baseband for both I and Q bands?
            
            # 2b. Code wipeoff and summation
            #  Here we run a brief for-loop to obtain the early, late, and promt correlator outputs. 
            epl_correlations = []
            for chip_delay in [epl_chip_spacing, 0, -epl_chip_spacing]:                                             # Early, prompt, and late correlator timing
                chip_indices = (code_phase + chip_delay + code_rate * block_time).astype(int) % L1CA_CODE_LENGTH    # indices of relevant chips in the sample
                code_samples = 1 - 2 * code_seq[chip_indices]                                                       # Code samples, ranging from [-1,1]
                epl_correlations.append(numpy.mean(block_wo_carrier * code_samples))                                # performing the correlation
            early, prompt, late = epl_correlations # result of the correlation
            earlys.append(early)
            prompts.append(prompt)
            lates.append(late)

            
            # 3. Use discriminators to estimate state errors
         
            # 3a. Compute code phase error using early-minus-late discriminator. This is based off of Lecture 06, slide 7
            code_phase_error = epl_chip_spacing * (abs(early) - abs(late)) / (abs(early) + abs(late) + 2*abs(prompt))
                        
            unfiltered_code_phase = code_phase + code_phase_error

            # 3b. Compute phase error (in cycles) using appropriate phase discriminator
            # (I know greek letters are typically in radians, but make sure `delta_theta` is in cycles)
            delta_theta = numpy.arctan(prompt.imag / prompt.real) / (2*pi)
            
            # Note: the phase error `delta_theta` is actually equal to:
            # `carr_phase_error + integration_time + doppler_error / 2`
            # for a 2nd-order PLL, but we'll only define "unfiltered" carrier phase for our outputs.
            carr_phase_error = delta_theta
            unfiltered_carr_phase = carr_phase + delta_theta
            
            
            # 4. Apply loop filters to reduce noise in state error estimates
            
            # 4a. Filter code phase error to reduce noise
            #  We implement the DLL filter by updating code phase in proportion to code phase 
            # dicriminator output.  The result has the equivalent response of a 1st-order DLL filter
            filtered_code_phase_error = 4 * integration_time * DLL_bandwidth * code_phase_error
            
            filtered_code_phase = code_phase + filtered_code_phase_error

            # 4b. Filter carrier phase error to reduce noise
            #  We implement the PLL filter by updating carrier phase and frequency in proportion to
            # the phase discriminator output in a way that has the equivalent response to a 2nd-order
            # PLL.
            xi = 1 / sqrt(2)
            omega_n = PLL_bandwidth / .53
            filtered_carr_phase_error = (2 * xi * omega_n * integration_time - 3 / 2 * omega_n**2 * integration_time**2) * delta_theta
            filtered_doppler_error = omega_n**2 * integration_time * delta_theta
            
            filtered_carr_phase = carr_phase + filtered_carr_phase_error
            filtered_doppler = doppler + filtered_doppler_error

            
            # 5. Save our tracking loop outputs
            outputs['sample_index'][block_index] = sample_index
            outputs['early'][block_index] = early
            outputs['prompt'][block_index] = prompt
            outputs['late'][block_index] = late
            outputs['code_phase'][block_index] = code_phase
            outputs['unfiltered_code_phase'][block_index] = unfiltered_code_phase
            outputs['filtered_code_phase'][block_index] = filtered_code_phase
            outputs['carr_phase'][block_index] = carr_phase
            outputs['unfiltered_carr_phase'][block_index] = unfiltered_carr_phase
            outputs['filtered_carr_phase'][block_index] = filtered_carr_phase
            outputs['doppler'][block_index] = doppler
            outputs['filtered_doppler'][block_index] = filtered_doppler

            
            # 6. Propagate state to next time epoch
            
            # As part of this step, we apply carrier-aiding by adjusting `code_rate` based on Doppler
            code_rate = L1CA_CODE_RATE * (1 + doppler / L1CA_CARRIER_FREQ)
            
            # First we adjust the nominal time step to go to start of next desired chip
            if block_index+1 < N_blocks:
                target_code_phase = (block_index + 1) * block_length_chips
                sample_step = int((target_code_phase - filtered_code_phase) * source_params['samp_rate'] / code_rate)
                time_step = sample_step / source_params['samp_rate']
            
            # Then we update the states and sample index accordingly
            code_phase = filtered_code_phase + code_rate * time_step
            carr_phase = filtered_carr_phase + (inter_freq + doppler) * time_step
            doppler = filtered_doppler
            
            sample_index += sample_step

    for key in output_keys:
        outputs[key] = outputs[key][:block_index]
    outputs['prn'] = prn
    ### getting correlator magnitudes for open-loop data bits ###
    outputs['early'] = numpy.array(earlys)
    outputs['prompt'] = numpy.array(prompts)
    outputs['late'] = numpy.array(lates)
    outputs['DLL_bandwidth'] = DLL_bandwidth
    outputs['PLL_bandwidth'] = PLL_bandwidth
    outputs['N_integration_code_periods'] = N_integration_code_periods
    outputs['integration_time'] = integration_time
    outputs['time'] = outputs['sample_index'] / source_params['samp_rate']
    outputs['epl_chip_spacing'] = epl_chip_spacing 
    
    return outputs

#%% track_GPS_L1CA_signal_open function
def track_GPS_L1CA_signal_open(prn, source_params, model_time, model_code_phase, model_doppler, closed_correlator=None, **kwargs):
    '''    
    Given a PRN, acquires and tracks the corresponding GPS L1CA signal.
    
    Inputs:
        `prn` -- the PRN of the GPS satellite to acquire and track
        `source_params` -- dict containing information about the file source
        'model_time' -- SECONDS into the FILE (NOT TOW)
        
        `N_integration_code_periods` -- number of code periods (default 1) over which to coherently integrate when tracking
        `epl_chip_spacing` -- spacing of the EPL correlators in units of chips (default 0.5)

        NEW inputs required for open-loop aquisition: 
            'signal_model' -- a dict containing a pre-computed model of the signal
            `task2` -- indication of whether or not to remove data bits using closed-loop tracking
            ** NOT NEEDED? ** 'nav_soln' -- a dict containing the navigation soln, rx/tx timestamps, ECEF coords, drift, etc        

        OLD closed-loop inputs that are NOT NEEDED for open-loop aquisition: 
            `DLL_bandwidth` -- the bandwidth of the DLL (delay-locked loop) loop filter in Hz (default 5)
            `PLL_bandwidth` -- the bandwidth of the PLL (phase-locked loop) filter in Hz (default 20)
    
    Notes:
    
    In order to avoid data bit transitions, our tracking loop will correlate over an integer number of the CA code periods.
    Nominally the code period is 1 ms, which at a sampling rate of 5 MHz comes out to 5000 samples.  However, due to Doppler
    expansion/compression, the actual number of samples will be slightly above or slightly below 5000.  We adjust the time
    step accordingly in the tracking loop, but the nominal time step is sufficient for designing our loop filter.

    '''       
    # 0. Here we set up the tracking loop
    #  Any computations we can do outside the main loop will speed up our code, improve our own lives,
    # and be better for the planet.
    
    # The `sample_loader` object will help us load samples from the binary file stream
    sample_loader = SampleLoader(source_params['samp_rate'], source_params['bit_depth'],
                                 source_params['is_signed'], source_params['is_integer'],
                                 source_params['is_complex'], source_params['i_lsn'])
    
    # Here we define the "tracking block size" as an integer multiple of the L1CA code period.
    # Nominally, the tracking block size will be some multiple of 1 millisecond, and is equal to the
    # coherent integration time as well as the nominal tracking loop update time step.  Due to code
    # expansion / compression, the actual tracking loop update time step will be expanded or
    # compressed by a small amount.  However, this expansion / compression will not affect the design
    # of our estimators or loop filters.
    N_integration_code_periods = kwargs.get('N_integration_code_periods', 1)
    block_length_chips = L1CA_CODE_LENGTH * N_integration_code_periods
    block_duration = block_length_chips / L1CA_CODE_RATE
    integration_time = block_duration

    # Here we define a time array that we use for sampling our reference signal
    N_block_samples = int(integration_time * source_params['samp_rate']) # number of samples per block
    sample_time = arange(N_block_samples) / source_params['samp_rate']
    N_blocks = len(model_time)
    if type(closed_correlator) != 'NoneType':
        N_blocks = len(closed_correlator)

    
    # The EPL correlation delay spacing controls the sensitivity of the DLL to noise vs. multipath.
    # EPL stands for early, prompt, and late (correlators)
    epl_chip_spacing = kwargs.get('epl_chip_spacing', 0.5)

    # Here we preallocate our outputs
    output_keys = ['sample_index', 'early', 'prompt', 'late',
                   'code_phase', 'unfiltered_code_phase',
                   'carr_phase','doppler', 
                   'nav_bits', 
                   'disc_q2', 'disc_q4','closed_corr']
    output_dtypes = [int, complex, complex, complex,
                     float, float, float, float, 
                     float, float, float, complex]
    outputs = {key: numpy.zeros(N_blocks, dtype=dtype) for key, dtype in zip(output_keys, output_dtypes)}

    # Compute the intermediate frequency
    inter_freq = L1CA_CARRIER_FREQ - source_params['center_freq']
    
    # Get the appropriate PRN code sequence. Comes in [0,1]
    code_seq = get_GPS_L1CA_code_sequence(prn)
        
    # Open the IF sample file
    carr_phase = 0
    disc = []
    with open(source_params['filepath'], 'rb') as f:
        for block_index in range(N_blocks):
            sample_index = round(model_time[block_index] * source_params['samp_rate'])
            
            # 0. end the loop if you're at the end of the file
            if sample_index + N_block_samples >= source_params['file_length']:
                break
            print('\r {0: >4.1f}'.format(sample_index / source_params['samp_rate']), end='')   

            # 1. Get the next block of samples of the data and model
            block = sample_loader.generate_sample_block(f, sample_index, N_block_samples)
                        
            # get the next set of model parameters
            code_phase = model_code_phase[block_index]
            doppler = model_doppler[block_index]
            code_rate = L1CA_CODE_RATE * (1 + doppler / L1CA_CARRIER_FREQ)
  
            # 2. Reference Generation and Correlation
            #  For efficiency, this step is broken down into carrier wipeoff, code wipeoff, and
            # summation.  This process is equivalent to generating complete references and
            # correlating them with our IF samples.
            
            # By this time, the referenece is generated, and is required for the wipeoff operations.

            # 2a. Wipeoff carrier
            # This is mixing the recorded signal down to baseband 
            phi = 2 * pi * (inter_freq + doppler) * sample_time + carr_phase * 2 * pi
            carrier_conj = numpy.cos(phi) - 1j * numpy.sin(phi)     # conjugate of the carrier
            block_wo_carrier = block * carrier_conj                 # Mixing down to baseband for both I and Q bands
            
            # 2b. Code wipeoff and summation
            #  Here we run a brief for-loop to obtain the early, late, and promt correlator outputs. 
            # could add flexibility of epl range AND step
            epl_correlations = []
            epl_bits = []
            for chip_delay in [epl_chip_spacing, -epl_chip_spacing, 0 ]:                                             # Early, prompt, and late correlator timing
               # reference signal generation
               chip_indices = (code_phase + chip_delay + code_rate * sample_time).astype(int) % L1CA_CODE_LENGTH    # indices of relevant chips in the sample
               code_samples = 1 - 2 * code_seq[chip_indices]                                                       # Code samples, ranging from [-1,1]
                                
               epl_correlations.append(numpy.mean(block_wo_carrier * code_samples))                                # performing the correlation
            early, late, prompt = epl_correlations # result of the correlation
            
            # Finding the navigation bits   
            costas_q2 = (numpy.arctan(closed_correlator[block_index].imag / closed_correlator[block_index].real))
            bits = round(abs(costas_q2/(pi/2)) - abs(numpy.angle(closed_correlator[block_index])/pi))
            
            # removing the data bits
            prompt_noData = ((bits * 2) - 1)  * prompt 
            disc_q4 = (numpy.arctan2(prompt_noData.imag, prompt_noData.real))
            disc_q2 = (numpy.arctan(prompt_noData.imag / prompt_noData.real))
            
            # 3. Use discriminators to estimate state errors. This step will not be a part of the open loop         
            # 3a. Compute code phase error using early-minus-late discriminator. This is based off of Lecture 06, slide 7           
            code_phase_error = epl_chip_spacing * (abs(early) - abs(late)) / (abs(early) + abs(late) + 2*abs(prompt))
            unfiltered_code_phase = code_phase + code_phase_error
            
            # carrier phase calculation
            if (block_index+1 < N_blocks):
                time_step = model_time[block_index + 1] - model_time[block_index]
                carr_phase += (inter_freq + doppler) * time_step 
            
            # 5. Save our tracking loop outputs
            outputs['sample_index'][block_index] = sample_index
            outputs['early'][block_index] = early
            outputs['prompt'][block_index] = prompt
            outputs['late'][block_index] = late
            outputs['code_phase'][block_index] = code_phase
            outputs['unfiltered_code_phase'][block_index] = unfiltered_code_phase
            outputs['carr_phase'][block_index] = carr_phase
            outputs['doppler'][block_index] = doppler
            outputs['nav_bits'][block_index] = bits
            outputs['disc_q2'][block_index] = disc_q2
            outputs['disc_q4'][block_index] = disc_q4
            
    for key in output_keys:
        outputs[key] = outputs[key][:block_index]
    outputs['prn'] = prn
    outputs['N_integration_code_periods'] = N_integration_code_periods
    outputs['integration_time'] = integration_time
    outputs['time'] = outputs['sample_index'] / source_params['samp_rate']
    outputs['epl_chip_spacing'] = epl_chip_spacing 
    outputs['closed_corr'] = closed_correlator
    
    return outputs

#%% Loading the signal model, navigation solution, and DTU18 MSS model.
# loading signal model
signalModel_filepath = './Data/haleakala_20210611_160000_RX7_signal_model.mat'
signalModel_in = loadmat(signalModel_filepath, squeeze_me=True)
signalModel_keys = ['az_dlos', 'az_sp', 'doppler_D', 'doppler_R', 'el_dlos',
                    'el_sp', 'gpsweek', 'sp_lat', 'sp_lon', 'sp_mss', 'tau_D',
                    'tau_R', 'timeVec']
signalModel = {}  #{key: [0] for key in signalModel_keys}
for keys in signalModel_keys:
    signalModel[keys] = signalModel_in[keys]
del signalModel_in

# please don't load DTU18. It is very large. 
# DTU18_filepath = './Data/dtu18.mat'
# DTU18 = loadmat(DTU18_filepath)
    
#%% Choose IF data file and appropriate data parameters
data_filepath = './Data/haleakala_20210611_160000_RX7.dat'

# The file contains complex 8-bit samples, where the first byte is the real component
# The sampling rate is 5 MHz and the front-end center frequency is at GPS L1
source_params_info = {
        'samp_rate': 5e6,
        'center_freq': 1.57542e9,
        'bit_depth': 8,
        'is_complex': True,
        'is_signed': True,
        'is_integer': True,
        'i_lsb': True,
        'filepath': data_filepath,
    }
source_params = get_file_source_info(**source_params_info)

file_start_time_dt = datetime(2021, 6, 11, 16)
file_start_time_gpst = dt2gpst(file_start_time_dt)

#%%  Choose PRN and tracking parameters,
prn = 5
offset = 0

# model data
# model_time = signalModel['timeVec'] - signalModel['timeVec'][0]
# model_time =  model_time[offset:]
model_time = numpy.arange(0,60-0.001,0.001) 
nominal_code_phase = model_time * L1CA_CODE_RATE
model_doppler = signalModel['doppler_D'][offset:,prn-1]
model_code_phase = signalModel['tau_D'][offset:,prn-1] + nominal_code_phase
N_integration_code_periods = 1

# Creating output folder
output_dir = './tracking-output/'
os.makedirs(output_dir, exist_ok=True)
epl_chip_spacing = 0.5

#%% Closed loop tracking 
# Acquire
c_acq, f_acq, n_acq = acquire_GPS_L1CA_signal(data_filepath, source_params, prn, 0)

# Track: closed and open
outputs_c = track_GPS_L1CA_signal_closed(prn, source_params, 0, n_acq['code_phase'], f_acq['doppler'],
                                         N_integration_code_periods=N_integration_code_periods,
                                         DLL_bandwidth=5, PLL_bandwidth=20, epl_chip_spacing=epl_chip_spacing)
prompt_c = outputs_c['prompt']

#%% Open loop tracking(and data bit aquisition, task 2)
outputs_o = track_GPS_L1CA_signal_open(prn, source_params, model_time, model_code_phase, model_doppler, 
                                       task2=True, quadrant=2, closed_correlator=prompt_c)

output_filename = 'PRN-{0:02}_N-int-{1:02}_chpWd-{2:02}_OLR_Q2.mat'.format(
    prn, N_integration_code_periods,epl_chip_spacing)
output_filepath = os.path.join(output_dir, output_filename)
with h5py.File(output_filepath, 'w') as f:
    write_dict_to_hdf5(outputs_o, f)

#%% Loading in the navigation data to produce models:
navData_filepath = './Data/haleakala_20210611_160000_RX7_nav.mat'
navData_in = loadmat(navData_filepath, squeeze_me=True)
navData_keys = ['__header__', '__version__', '__globals__', 'Rx_Clk_Bias', 
                'Rx_Clk_Drift', 'Rx_TimeStamp', 'Rx_Vx', 'Rx_Vy', 'Rx_Vz', 
                'Rx_X', 'Rx_Y', 'Rx_Z', 'Rx_height', 'Rx_lat', 'Rx_lon']
navData = {}  #{key: [0] for key in signalModel_keys}
for keys in navData_keys:
    navData[keys] = navData_in[keys]
del navData_in

#%% Producing model data

# Starting with constants
c = 3e8 # m/s
fL1 = 1.57542e9
# Then getting variables we need from the appropriate files
tu = navData['Rx_TimeStamp']
Slat,Slon,Salt = signalModel['sp_lat'][offset:,prn-1], signalModel['sp_lon'][offset:,prn-1], signalModel['sp_mss'][offset:,prn-1]
Rx, Ry, Rz = navData['Rx_X'], navData['Rx_Y'], navData['Rx_Z']
# Doing linear fits on the receiver location values so they're not as noisy:
### Starting with Rx
x_lin = numpy.array([stats.linregress(tu,Rx).slope,stats.linregress(tu,Rx).intercept])
Rx_lin = x_lin[0]*tu+x_lin[1]
y_lin = numpy.array([stats.linregress(tu,Ry).slope,stats.linregress(tu,Ry).intercept])
Ry_lin = y_lin[0]*tu+y_lin[1]
z_lin = numpy.array([stats.linregress(tu,Rz).slope,stats.linregress(tu,Rz).intercept])
Rz_lin = z_lin[0]*tu+z_lin[1]

Vx, Vy, Vz = navData['Rx_Vx'], navData['Rx_Vy'], navData['Rx_Vz']
ClockDrift = navData['Rx_Clk_Drift']
# Linear fitting the drift because the values are kinda strange:
d_lin = numpy.array([stats.linregress(tu,ClockDrift).slope,stats.linregress(tu,ClockDrift).intercept])
CD_lin = d_lin[0]*tu+d_lin[1]
ClockBias = navData['Rx_Clk_Bias']

# Here we have a function to convert LLA coordinates to ECEF 
## Source: https://gis.stackexchange.com/questions/230160/converting-wgs84-to-ecef-in-python
import pyproj
def gps_to_ecef_pyproj(lat, lon, alt):
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    x, y, z = pyproj.transform(lla, ecef, lon, lat, alt, radians=False)
    return x, y, z

# Converting/calculating values we need:
Sx, Sy, Sz = gps_to_ecef_pyproj(Slat, Slon, Salt)
GeoRange = numpy.sqrt((Sx - Rx_lin)**2 + (Sy - Ry_lin)**2 + (Sz - Rz_lin)**2)
Range = GeoRange - (ClockBias * c) - CD_lin

# Finally calculating the models:
tau_C = (tu - (abs(GeoRange)/c) + signalModel['timeVec'])
doppler_C = numpy.diff(Range) * 2 * pi * fL1 / c

#%% Plotting the models/difference:
import matplotlib.pyplot as plt
plt.scatter(tu[:-1],doppler_C,s=1)
plt.plot(tu,signalModel['doppler_D'][offset:,prn-1],c='red')
#plt.xlim(10000,60000)

#%% Prepping for the challenge question by loading in the data file:
DTU18_filepath = './Data/dtu18.mat'
DTU18 = loadmat(DTU18_filepath, squeeze_me=True)['dtu18']
DTU18_lon = DTU18['lon']
DTU18_lat = DTU18['lat']
DTU18_mss = DTU18['mss']