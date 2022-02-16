'''
gpsl1ca.py

Routines for generating the GPS L1CA signal

@author Brian Breitsch
@email brian.breitsch@colorado.edu
'''

import numpy
from numpy import zeros, arange, var, mean, unravel_index, unwrap, angle, exp, conj
from numpy.fft import fft, ifft
from .mls import generate_mls
from .file_source import SampleLoader
from scipy.constants import pi, speed_of_light

L1CA_CODE_LENGTH = 1023
L1CA_CODE_RATE = 1.023E6
L1CA_DATA_SYMBOL_RATE = 50
L1CA_CARRIER_FREQ = 1.57542e9

'''
(svid, prn, ca_phase_select, x2_phase_select, ca_code_delay, p_code_delay, first_10_chips_ca, first_12_chips_p)
Tuple struct to store data from Table 3-I of the IS-GPS 200 
specification, which contains code phase assignment information for GPS L1 signal.

`ca_phase_select` is a 2-tuple in this structure.

`first_12_chips_p`, `first_10_chips_ca` are represented in octal in the table,
but should just be integer types in this structure.

Note that SVID and PRN numbers differ only for SVIDs 65-69.
'''
L1_CODE_PHASE_ASSIGNMENTS = { 
    1 : (1, 1, (2, 6), 1, 5, 1, 1440, 4444),
    2 : (2, 2, (3, 7), 2, 6, 2, 1620, 4000),
    3 : (3, 3, (4, 8), 3, 7, 3, 1710, 4333),
    4 : (4, 4, (5, 9), 4, 8, 4, 1744, 4377),
    5 : (5, 5, (1, 9), 5, 17, 5, 1133, 4355),
    6 : (6, 6, (2, 10), 6, 18, 6, 1455, 4344),
    7 : (7, 7, (1, 8), 7, 139, 7, 1131, 4340),
    8 : (8, 8, (2, 9), 8, 140, 8, 1454, 4342),
    9 : (9, 9, (3, 10), 9, 141, 9, 1626, 4343),
    10 : (10, 10, (2, 3), 10, 251, 10, 1504, 4343),
    11 : (11, 11, (3, 4), 11, 252, 11, 1642, 4343),
    12 : (12, 12, (5, 6), 12, 254, 12, 1750, 4343),
    13 : (13, 13, (6, 7), 13, 255, 13, 1764, 4343),
    14 : (14, 14, (7, 8), 14, 256, 14, 1772, 4343),
    15 : (15, 15, (8, 9), 15, 257, 15, 1775, 4343),
    16 : (16, 16, (9, 10), 16, 258, 16, 1776, 4343),
    17 : (17, 17, (1, 4), 17, 469, 17, 1156, 4343),
    18 : (18, 18, (2, 5), 18, 470, 18, 1467, 4343),
    19 : (19, 19, (3, 6), 19, 471, 19, 1633, 4343),
    20 : (20, 20, (4, 7), 20, 472, 20, 1715, 4343),
    21 : (21, 21, (5, 8), 21, 473, 21, 1746, 4343),
    22 : (22, 22, (6, 9), 22, 474, 22, 1763, 4343),
    23 : (23, 23, (1, 3), 23, 509, 23, 1063, 4343),
    24 : (24, 24, (4, 6), 24, 512, 24, 1706, 4343),
    25 : (25, 25, (5, 7), 25, 513, 25, 1743, 4343),
    26 : (26, 26, (6, 8), 26, 514, 26, 1761, 4343),
    27 : (27, 27, (7, 9), 27, 515, 27, 1770, 4343),
    28 : (28, 28, (8, 10), 28, 516, 28, 1774, 4343),
    29 : (29, 29, (1, 6), 29, 859, 29, 1127, 4343),
    30 : (30, 30, (2, 7), 30, 860, 30, 1453, 4343),
    31 : (31, 31, (3, 8), 31, 861, 31, 1625, 4343),
    32 : (32, 32, (4, 9), 32, 862, 32, 1712, 4343),
    33 : (65, 33, (5, 10), 33, 863, 33, 1745, 4343),
    34 : (66, 34, (4, 10), 34, 950, 34, 1713, 4343),
    35 : (67, 35, (1, 7), 35, 947, 35, 1134, 4343),
    36 : (68, 36, (2, 8), 36, 948, 36, 1456, 4343),
    37 : (69, 37, (4, 10), 37, 950, 37, 1713, 4343),}


def generate_GPS_L1CA_code_sequence(prn):
    '''Generates GPS L1 CA PRN code for given PRN.
    
    Parameters
    ----------
    prn : int
        the PRN of the signal/satellite

    Returns
    -------
    output : ndarray of shape(1023,)
        the complete code sequence
    '''
    ps = L1_CODE_PHASE_ASSIGNMENTS[prn][2]
    g1 = generate_mls(10, [2, 9], [9])
    g2 = generate_mls(10, [1, 2, 5, 7, 8, 9], [ps[0] - 1, ps[1] - 1])
    return (g1 + g2) % 2

_GPS_L1CA_code_sequences_ = {}
def get_GPS_L1CA_code_sequence(prn):
    '''
    Returns the code sequence corresponding to the given PRN
    '''
    if prn not in _GPS_L1CA_code_sequences_:
        _GPS_L1CA_code_sequences_[prn] = generate_GPS_L1CA_code_sequence(prn)
    return _GPS_L1CA_code_sequences_[prn]

def sample_GPS_L1CA_code(prn, chips, data_bits=None, data_bit_start_chip=0):
    '''
    Samples and returns the code sequence for the given PRN and code phase
    sequence `chips`
    ----------------------------------------------------------------------------
    '''
    code_seq = get_GPS_L1CA_code_sequence(prn)
    code_samples = code_seq[chips.astype(int) % L1CA_CODE_LENGTH]
    if data_bits is not None:
        data_phase = (chips - data_bit_start_chip) * L1CA_DATA_SYMBOL_RATE / L1CA_CODE_RATE
        data_samples = data_bits[data_phase.astype(int) % len(data_bits)]
        code_samples = (code_samples + data_samples) % 2
    return code_samples

def sample_GPS_L1CA_signal(prn, time, chip0, chip_rate, phi0, phi_rate, phi_accel=0, data_bits=None, data_bit_start_chip=0):
    '''
    Samples and returns the complex baseband signal sequence given the PRN and signal
    phase parameters.
    ----------------------------------------------------------------------------
    '''
    chips = chip0 + chip_rate * time
    phi = phi0 + phi_rate * time + 0.5 * phi_accel * time**2
    code_samples = sample_GPS_L1CA_code(prn, chips, data_bits=data_bits, data_bit_start_chip=data_bit_start_chip)
    C = 1 - 2 * code_samples
    return C * exp(1j * phi)

def coarse_acquire_GPS_L1CA_signal(samples, samp_rate, center_freq, prn, L_coh=1, N_ncoh=1, vrel_max=1000, vrel_min=-1000, return_correlation=False, **kwargs):
    '''
    Acquires the Doppler frequency and code phase paramters of a signal.
    --------------------------------------------------------------------------------------------------------------------
    `samples` -- the sample buffer; acquisition will be at the start of the buffer
    `samp_rate` -- the buffer sampling rate
    `center_freq` -- the center frequency of the front end
    `prn` -- satellite PRN
    `L_coh` -- the number of code_periods to coherently integrate
    `N_ncoh` -- the number of coherent blocks to noncoherently integrate
    `vrel_max` -- the maximum relative velocity (m/s), used for generating Doppler search bins
    `vrel_max` -- the minimum relative velocity (m/s), used for generating Doppler search bins
    `return_correlation` -- whether to include the correlation matrix in output

    Returns:
    `outputs` -- dict with `doppler`, `code_phase`, `snr`, `cn0
    '''    
    T_coh = L_coh * L1CA_CODE_LENGTH / L1CA_CODE_RATE
    N_coh = int(T_coh * samp_rate)
    N = N_ncoh * N_coh
    
    dopp_max = vrel_max * L1CA_CARRIER_FREQ / speed_of_light
    dopp_min = vrel_min * L1CA_CARRIER_FREQ / speed_of_light
    doppler_bins = arange(dopp_min, dopp_max, 1 / T_coh)
    
    # no data bit handling in coarse acquisition -- may necessitate non-coherent integration for certain coherent integration periods
    correlation = zeros((len(doppler_bins), N_coh))
    fft_blocks = fft(samples[:N].reshape((N_ncoh, N_coh)), axis=1)
    samp_var = numpy.var(samples.real[:N])
    
    time = arange(N_coh) / samp_rate
    for i, doppler in enumerate(doppler_bins):
        code_rate = L1CA_CODE_RATE * (1 + doppler / L1CA_CARRIER_FREQ)
        phi_rate = 2 * pi * (L1CA_CARRIER_FREQ - center_freq + doppler)
        s_ref = sample_GPS_L1CA_signal(prn, time, 0, code_rate, 0, phi_rate)
        correlation[i] = numpy.sum(abs(ifft(conj(fft_blocks) * fft(s_ref)[None, :])), axis=0)
    
    # Compute number of samples in one code period
    N_1cp = int(L1CA_CODE_LENGTH / L1CA_CODE_RATE * samp_rate)
    corr = correlation[:, :N_1cp]
    dopp_bin, sample_bin = unravel_index(corr.argmax(), corr.shape)
    max_val = corr[dopp_bin, sample_bin]
    noise_var = (var(corr) * corr.size - max_val**2) / (corr.size - 1)  # TODO better way of computing noise var?
    snr = max_val**2 / noise_var
    doppler = doppler_bins[dopp_bin]
    
    # Calculate chip phase from sample phase
    code_phase = sample_bin / N_1cp * L1CA_CODE_LENGTH
    cnr = snr / (N_ncoh * T_coh)
    
    outputs = {
        'snr': snr,
        'cnr': cnr,
        'doppler': doppler,
        'code_phase': code_phase,
        'noise_var': noise_var
    }
    if return_correlation:
        outputs['correlation'] = corr
        outputs['doppler_bins'] = doppler_bins
        outputs['code_phase_bins'] = arange(N_1cp) / samp_rate * L1CA_CODE_RATE
    
    return outputs

def fine_acquire_GPS_L1CA_signal(samples, samp_rate, center_freq, prn, code_phase_acq, doppler_acq, L_coh, N_blks):
    '''
    This function is performed after coarse acquisition of the L1CA signal.  It refines the Doppler estimate and
    estimates the carrier phase offset.
    --------------------------------------------------------------------------------------------------------------------
    '''
    code_rate = L1CA_CODE_RATE * (1 + doppler_acq / L1CA_CARRIER_FREQ)
    N_samples = N_blks * L_coh * L1CA_CODE_LENGTH / code_rate * samp_rate
    
    first_sample_index = int((L1CA_CODE_LENGTH - code_phase_acq) / code_rate * samp_rate)
    samples = samples[first_sample_index:]
    
    phi_rate = 2 * pi * (L1CA_CARRIER_FREQ - center_freq + doppler_acq)    
    N = int(L_coh * L1CA_CODE_LENGTH / code_rate * samp_rate)
    
    samples = samples[:N * N_blks]
    time = arange(N * N_blks) / samp_rate
    s_ref = sample_GPS_L1CA_signal(prn, time, 0, code_rate, 0, phi_rate)
    correlation_blocks = (samples * conj(s_ref)).reshape((N_blks, N)).sum(axis=-1)
    
    delta_phase = numpy.angle(correlation_blocks[1:] / correlation_blocks[:-1])
    indices = numpy.where((delta_phase - mean(delta_phase))**2 < var(delta_phase))[0]
    slope = mean(delta_phase[indices])
    doppler_delta = slope * samp_rate / (N * 2 * pi)
    phases = unwrap(angle(correlation_blocks))
    t_blks = arange(N_blks) * N / samp_rate
    phase_slope, phi0 = numpy.polyfit(t_blks, phases, 1)
    
    outputs = {
        'correlation': correlation_blocks,
        'phases': phases,
        't_blks': t_blks,
        'phi0': phi0,
        'doppler_delta': doppler_delta,
        'doppler': doppler_acq + doppler_delta,
    }
    return outputs
    
def acquire_GPS_L1CA_data_bit_phase(samples, samp_rate, center_freq, prn, code_phase_acq, doppler_acq, N_blks=500):
    '''
    This function is performed after coarse acquisition (and optionally, after fine acquisition) of the L1CA signal.
    It finds the location of navigation data bit edges to provide a more absolute code phase estimate so that tracking
    or other processes can know when potential bit transitions happen.
    --------------------------------------------------------------------------------------------------------------------
    '''      
    L_coh = 1
    # There are 20 1ms blocks per data bit, so N_blks needs to be a multiple of 20
    assert(N_blks % 20 == 0)
    
    code_rate = L1CA_CODE_RATE * (1 + doppler_acq / L1CA_CARRIER_FREQ)
    N_samples = N_blks * L1CA_CODE_LENGTH / code_rate * samp_rate

    first_sample_index = int((L1CA_CODE_LENGTH - code_phase_acq) / code_rate * samp_rate)
    samples = samples[first_sample_index:]
    
    phi_rate = 2 * pi * (L1CA_CARRIER_FREQ - center_freq + doppler_acq)
    generate_reference = lambda time: sample_GPS_L1CA_signal(prn, time, 0, code_rate, 0, phi_rate)
    
    N = int(L_coh * L1CA_CODE_LENGTH / code_rate * samp_rate)
    samples = samples[:N * N_blks]
    time = arange(N * N_blks) / samp_rate
    s_ref = sample_GPS_L1CA_signal(prn, time, 0, code_rate, 0, phi_rate)
    correlation_blocks = (samples * conj(s_ref)).reshape((N_blks, N)).sum(axis=-1)
    
    corr = []
    for i in range(20):
        coh = numpy.sum(numpy.roll(correlation_blocks, i).reshape((-1, 20)), axis=1)
        corr.append(sum(abs(coh)))

    data_bit_phase = numpy.argmax(corr)
    # I know it seems weird at first but this is correct
    new_code_phase = code_phase_acq + (data_bit_phase - 1) * L1CA_CODE_LENGTH
    
    outputs = {
        'shift_correlation': corr,
        'data_bit_phase': data_bit_phase,
        'code_phase': new_code_phase
    }
    return outputs

def acquire_GPS_L1CA_signal(filepath, source_params, prn, start_sample, c_acq_L_coh=5, c_acq_N_ncoh=2, f_acq_L_coh=1, f_acq_N_blks=40, print_results=True):
    '''
    Helper function to run all three acquisition functions
    
    Returns:
        `c_acq, f_acq, n_acq` -- coarse acquisition, fine acquisition, and data bit alignment results
    '''
    # Get signal block for acquisition
    sample_loader = SampleLoader(source_params['samp_rate'], source_params['bit_depth'],source_params['is_signed'],
                                 source_params['is_integer'], source_params['is_complex'], source_params['i_lsn'])
    with open(source_params['filepath'], 'rb') as f:
        block = sample_loader.generate_sample_block(f, start_sample, int(1 * source_params['samp_rate']))
    
    c_acq = coarse_acquire_GPS_L1CA_signal(block, source_params['samp_rate'], source_params['center_freq'],
                    prn, c_acq_L_coh, c_acq_N_ncoh, vrel_max=1000, vrel_min=-1000, return_correlation=True)

    # Fine acquire
    f_acq = fine_acquire_GPS_L1CA_signal(block, source_params['samp_rate'], source_params['center_freq'], prn,
                         c_acq['code_phase'], c_acq['doppler'], f_acq_L_coh, f_acq_N_blks)

    # Nav bit synchronization
    n_acq = acquire_GPS_L1CA_data_bit_phase(block, source_params['samp_rate'], source_params['center_freq'], prn,
            c_acq['code_phase'], f_acq['doppler'], N_blks=500)
    
    if print_results:
        print('Code Phase: {0:3.3f} chips \tDoppler Freq: {1:3.3f} \t C/N0: {2:3.3f}'.format(
            c_acq['code_phase'], c_acq['doppler'], 10 * numpy.log10(c_acq['cnr'])))
        print('Phi0: {0:3.3f} rad \tDoppler Freq: {1:3.3f} \t Dopp. Delta: {2:3.3f}'.format(
            f_acq['phi0'], f_acq['doppler'], f_acq['doppler_delta']))
        print('Data Bit Phase: {0:3.3f}'.format(n_acq['data_bit_phase']))
    
    return c_acq, f_acq, n_acq
