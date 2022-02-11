import os, h5py, numpy
from datetime import datetime
from numpy import zeros, arange, nan, real, imag, conj, pi, sin, arctan, arctan2, angle, exp, sqrt, diff, pad, std
from utilities.gpsl1ca import get_GPS_L1CA_code_sequence, acquire_GPS_L1CA_signal, \
    L1CA_CODE_RATE, L1CA_CODE_LENGTH, L1CA_CARRIER_FREQ
from utilities.file_source import get_file_source_info, SampleLoader
from utilities.gpst import dt2gpst, gpst2dt
from utilities.hdf5_utils import write_dict_to_hdf5

# other libraries 
import math

#%% track_GPS_L1CA_signal definition
def track_GPS_L1CA_signal(prn, source_params, acq_sample_index, code_phase_acq, doppler_acq, **kwargs):
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
    outputs['DLL_bandwidth'] = DLL_bandwidth
    outputs['PLL_bandwidth'] = PLL_bandwidth
    outputs['N_integration_code_periods'] = N_integration_code_periods
    outputs['integration_time'] = integration_time
    outputs['time'] = outputs['sample_index'] / source_params['samp_rate']
    outputs['epl_chip_spacing'] = epl_chip_spacing 
    
    return outputs

#%% Choose IF data file and appropriate data parameters

# This should be the path to raw IF data file that you download from the class site
data_filepath = '../Data/haleakala_20210611_160000_RX7.dat'

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

#%%  Choose PRN and tracking parameters
prn = 2
challengeQuestion = True    # set to FALSE if the challenge question is not being investigated

# The following variables are chosen by us (Brenna and James)
N_integration_code_periods = 1
epl_chip_spacing = 0.5 # (units: chips)

# challenge question: setting BWs to extreme values
if (challengeQuestion):    
    # by having the same number of steps in both parameters, each file can have 
    # a unique BW in both areas. 
    BW_steps = 5    # number of BW to investigate
    
    DLL_BW_start = 0.1    # start of BW range
    DLL_BW_stop = 1000.1    # stop of BW range
    
    PLL_BW_start = 0.1    # start of BW range
    PLL_BW_stop = 1000.1    # stop of BW range
    
    # creating BW vectors
    DLL_BW = numpy.linspace(DLL_BW_start,DLL_BW_stop,BW_steps)
    PLL_BW = numpy.linspace(PLL_BW_start,PLL_BW_stop,BW_steps)
    BW_all = numpy.transpose([DLL_BW, PLL_BW])
else:
    # if not doing the challenge question, only do 1 set of BW
    DLL_BW = [5]  # (units: Hz)
    PLL_BW = [20] # (units: Hz)
    

# Creating output folder
output_dir = './tracking-output/'
if (challengeQuestion):
    output_dir = './tracking-output/challengeQuestion_G{:02}/'.format(prn)
os.makedirs(output_dir, exist_ok=True)

# Recording output data 
for DLL_bandwidth in DLL_BW:
    for PLL_bandwidth in PLL_BW:
        # Acquire
        c_acq, f_acq, n_acq = acquire_GPS_L1CA_signal(data_filepath, source_params, prn, 0)

        # Track
        outputs = track_GPS_L1CA_signal(prn, source_params, 0, n_acq['code_phase'], f_acq['doppler'],
            N_integration_code_periods=N_integration_code_periods,
            DLL_bandwidth=DLL_bandwidth, PLL_bandwidth=PLL_bandwidth, epl_chip_spacing=epl_chip_spacing)

        output_filename = 'PRN-{0:02}_N-int-{1:02}_DLL-BW-{2:02}_PLL-BW-{3:02}_chpWd-{4:02}.mat'.format(
            prn, N_integration_code_periods, DLL_bandwidth, PLL_bandwidth,epl_chip_spacing)
        output_filepath = os.path.join(output_dir, output_filename)
        with h5py.File(output_filepath, 'w') as f:
            write_dict_to_hdf5(outputs, f)

