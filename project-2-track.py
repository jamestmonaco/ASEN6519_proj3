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

#%% track_GPS_L1CA_signal_open function
def track_GPS_L1CA_signal_open(prn, source_params, model_time, model_code_phase, model_doppler, **kwargs):
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
            ** NOT NEEDED? ** 'nav_soln' -- a dict containing the navigation soln, rx/tx timestamps, ECEF coords, drift, etc        

        OLD closed-loop inputs that are NOT NEEDED for open-loop aquisition: 
            `DLL_bandwidth` -- the bandwidth of the DLL (delay-locked loop) loop filter in Hz (default 5)
            `PLL_bandwidth` -- the bandwidth of the PLL (phase-locked loop) filter in Hz (default 20)
    
    Notes:
    
    In order to avoid data bit transitions, our tracking loop will correlate over an integer number of the CA code periods.
    Nominally the code period is 1 ms, which at a sampling rate of 5 MHz comes out to 5000 samples.  However, due to Doppler
    expansion/compression, the actual number of samples will be slightly above or slightly below 5000.  We adjust the time
    step accordingly in the tracking loop, but the nominal time step is sufficient for designing our loop filter.
    
    James' Notes on confusing things(tm): 
        * Delay locked loops (DLLs) are used to detect code phase (via code phase error of current estimate)
        * Phase locked loops (PLLs) are used to detect carrier phase and frequency (via error of current estimates)
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
    
     # The EPL correlation delay spacing controls the sensitivity of the DLL to noise vs. multipath.
    # EPL stands for early, prompt, and late (correlators)
    epl_chip_spacing = kwargs.get('epl_chip_spacing', 0.5)

    # Here we preallocate our outputs
    output_keys = ['sample_index', 'early', 'prompt', 'late',
                   'code_phase', 'unfiltered_code_phase',
                   'carr_phase','doppler']
    output_dtypes = [int, complex, complex, complex,
                     float, float,
                     float, float]
    outputs = {key: numpy.zeros(N_blocks, dtype=dtype) for key, dtype in zip(output_keys, output_dtypes)}

    # Compute the intermediate frequency
    inter_freq = L1CA_CARRIER_FREQ - source_params['center_freq']
    
    # Get the appropriate PRN code sequence. Comes in [0,1]
    code_seq = get_GPS_L1CA_code_sequence(prn)
    
    carr_phase = 0
    
    # Open the IF sample file
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
            for chip_delay in [epl_chip_spacing, -epl_chip_spacing, 0 ]:                                             # Early, prompt, and late correlator timing
               # reference signal generation
               chip_indices = (code_phase + chip_delay + code_rate * sample_time).astype(int) % L1CA_CODE_LENGTH    # indices of relevant chips in the sample
               code_samples = 1 - 2 * code_seq[chip_indices]                                                       # Code samples, ranging from [-1,1]
                                
               epl_correlations.append(numpy.mean(block_wo_carrier * code_samples))                                # performing the correlation
            early, late, prompt = epl_correlations # result of the correlation
             
            # 3. Use discriminators to estimate state errors. This step will not be a part of the open loop         
            # 3a. Compute code phase error using early-minus-late discriminator. This is based off of Lecture 06, slide 7           
            code_phase_error = epl_chip_spacing * (abs(early) - abs(late)) / (abs(early) + abs(late) + 2*abs(prompt))
            unfiltered_code_phase = code_phase + code_phase_error
            
            # carrier phase calculation
            if (block_index + 1 < N_blocks):
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
            

    for key in output_keys:
        outputs[key] = outputs[key][:block_index]
    outputs['prn'] = prn
    outputs['N_integration_code_periods'] = N_integration_code_periods
    outputs['integration_time'] = integration_time
    outputs['time'] = outputs['sample_index'] / source_params['samp_rate']
    outputs['epl_chip_spacing'] = epl_chip_spacing 
    
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

# # loading navigation soln
# navSoln_filepath = './Data/haleakala_20210611_160000_RX7_nav.mat'
# navSoln_in = loadmat(navSoln_filepath) 
# navSoln_keys = ['Rx_Clk_Bias', 'Rx_Clk_Drift','Rx_height', 'Rx_lat','Rx_lon',
#                 'Rx_TimeStamp','Rx_Vx','Rx_Vy','Rx_Vz','Rx_X','Rx_Y','Rx_Z']
# navSoln = {key: [0] for key in navSoln_keys}
# for key in navSoln_keys:
#     navSoln[key] = navSoln_in[key]
# del navSoln_in

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

#%%  Choose PRN and tracking parameters
prn = 5
offset = 0

# model data
# model_time = signalModel['timeVec'] - signalModel['timeVec'][0]
# model_time =  model_time[offset:]
model_time = numpy.arange(0,60-0.001,0.001)
 
nominal_code_phase = model_time * L1CA_CODE_RATE

model_doppler = signalModel['doppler_R'][offset:,prn-1]
model_code_phase = signalModel['tau_R'][offset:,prn-1] + nominal_code_phase

# The following variables are chosen by us (Brenna and James)
N_integration_code_periods = 1

# Creating output folder
output_dir = './tracking-output/'
os.makedirs(output_dir, exist_ok=True)
epl_chip_spacing = 0.5

# Acquire
# c_acq, f_acq, n_acq = acquire_GPS_L1CA_signal(data_filepath, source_params, prn, 0)

# Track
outputs = track_GPS_L1CA_signal_open(prn, source_params, model_time, model_code_phase, model_doppler)

output_filename = 'PRN-{0:02}_N-int-{1:02}_chpWd-{2:02}_OLR.mat'.format(
    prn, N_integration_code_periods,epl_chip_spacing)
output_filepath = os.path.join(output_dir, output_filename)
with h5py.File(output_filepath, 'w') as f:
    write_dict_to_hdf5(outputs, f)
    
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

#%% producing model:
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
GeoRange = numpy.sqrt((Sx-Rx_lin)**2 + (Sy-Ry_lin)**2 + (Sz-Rz_lin)**2)
Range = GeoRange - (ClockBias * c) - CD_lin

# Finally calculating the models:
tau_C = (signalModel['timeVec'] - (abs(GeoRange - Range)/c) - tu) * fL1
## doppler_C = ## Unsure how to calculate velocity for the satellite here??

#%% Plotting the models/difference:
import scipy.stats as stats
plt.plot(tau_C)
plt.plot(signalModel['tau_D'][offset:,prn-1])

#%% Testing my linear regression code because shit doesn't seem to be working rn
t = tu
A = Rz
b_lin = numpy.array([stats.linregress(t,A).slope,stats.linregress(t,A).intercept])
plt.plot(t,b_lin[0]*t+b_lin[1],c='red',zorder=2,alpha=0.75,label="Linear trend")
plt.plot(tu,Rz)



