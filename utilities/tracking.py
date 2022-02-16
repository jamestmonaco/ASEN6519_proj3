import numpy
from numpy import arange, array, sqrt, zeros, identity
from numpy.linalg import inv
from scipy.linalg import block_diag
from .kf_dynamics import compute_2nd_order_alpha_beta_from_xi_omega, compute_2nd_order_xi_omega_from_alpha_beta, \
    compute_2nd_order_steady_state_kalman_gain, compute_3rd_order_a_b_omega_from_alpha_beta_gamma, \
    compute_3rd_order_alpha_beta_gamma_from_a_b_omega, compute_3rd_order_steady_state_kalman_gain
from .kf_dynamics import create_2nd_order_Q, create_3rd_order_Q


def track_dataless(source, signal, f, code_phase_acq, phi_acq, doppler_acq, cnr_acq, **kwargs):
    '''
    Track a dataless signal
    `source` -- 
    `signal` --
    `f` -- handle to file stream
    `code_phase_acq` --
    `phi_acq` -- 
    `doppler_acq` -- 
    `cnr_acq` -- 
    `start_time` -- 
    `tracking_duration` -- 
    `epl_chip_spacing` --
    `chip_delta` -- 
    `chip_range` -- (start, stop) tuple indicating range of chips in `chip_offsets`
    `carrier_model_order` -- 
    `M_cnr_est` -- default 20
    `cnr_smoother_alpha` -- default .002
    '''
    
    samp_rate, f_center, generate_sample_block = (source[k] for k in ('samp_rate', 'f_center', 'generate_sample_block'))
    
    # Get file size
    block_length = int(1e-3 * samp_rate)
    block_duration = block_length / samp_rate
    samples_per_block = int(block_duration * samp_rate)
    assert(samples_per_block / samp_rate == block_duration)
    
    # Get start sample
    start_time = kwargs.get('start_time', 0)
    start_sample = int(start_time * samp_rate)
    assert(start_sample / samp_rate == start_time)
    
    # Get tracking duration / epochs
    tracking_duration = kwargs.get('tracking_duration', None)
    if tracking_duration is None:
        tracking_duration = source['file_duration'] - start_time
    Nt = int(tracking_duration / block_duration)
    
    time = arange(Nt) * block_duration
    
    Ny = 2  # dtau, dtheta
    
    epl_chip_spacing = kwargs.get('epl_chip_spacing', .5)
    chip_delta = kwargs.get('chip_delta', .25)
    assert(epl_chip_spacing % chip_delta == 0)
    chip_range = int(1.5 / chip_delta) * chip_delta
    chip_range = kwargs.get('chip_range', (-chip_range, chip_range))
    chip_offsets = arange(*chip_range, chip_delta)
    epl_indices = numpy.searchsorted(chip_offsets, [-epl_chip_spacing, 0, epl_chip_spacing])
    
    compute_sigma2_tau = lambda ti, cnr: epl_chip_spacing / (4 * cnr * ti) * (1 + 2 / ((2 - epl_chip_spacing) * cnr * ti))
    compute_sigma2_theta = lambda ti, cnr: 1 / (2 * cnr * ti) * (1 + 1 / (2 * cnr * ti))
    compute_R = lambda ti, cnr: diag((compute_sigma2_tau(ti, cnr), compute_sigma2_theta(ti, cnr)))

    # Always model 2nd-order code phase dynamics
    sig2_chip, sig2_chipdot = .05, .9
    Q_dll = create_2nd_order_Q(dt, sig2_chip, sig2_chipdot)
    
    carrier_model_order = kwargs.get('carrier_model_order', 2)
    assert(carrier_model_order == 2 or carrier_model_order == 3)
    
    if carrier_model_order == 2:
        # 2nd-order PLL
        Nx = 2 + 2  # code phase, code rate, phase, phase rate
        create_A = lambda dt: array([
                [1, dt, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, dt],
                [0, 0, 0, 1]])
        create_H = lambda ti: array([
            [1, 0, 0, 0],
            [0, 0, 1, ti / 2]
        ])
        sig2_phi, sig2_phidot = .05, 5
        create_Q = lambda dt: block_diag(, create_2nd_order_Q(dt, sig2_phi, sig2_phidot))
        
        omega_n_dll = 10
        omega_n_pll = 50
        xi = sqrt(2) / 2
        
        compute_steady_state_kalman_gain = lambda ti, omega_n_dll, omega_n_pll: block_diag(
            compute_2nd_order_steady_state_kalman_gain(ti, omega_n_dll, xi),
            compute_2nd_order_steady_state_kalman_gain(ti, omega_n_pll, xi),
        )
    elif carrier_model_order == 3:
        # 3rd-order PLL

        Nx = 5  # code phase, code rate, phase, phase rate
        create_A = lambda dt: array([
                [1, dt, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, dt, 0.5 * dt**2],
                [0, 0, 0, 1, dt],
                [0, 0, 0, 0, 1]])
        create_H = lambda ti: array([
            [1, 0, 0, 0, 0],
            [0, 0, 1, ti / 2, ti**2 / 6]
        ])
        sig2_phi, sig2_phidot, sig2_phiddot = .05, 1, .2
        create_Q = lambda dt: block_diag(create_2nd_order_Q(dt, sig2_chip, sig2_chipdot), create_3rd_order_Q(dt, sig2_phi, sig2_phidot, sig2_phiddot))
        
        omega_n_dll = 10
        omega_n_pll = 50
        xi = sqrt(2) / 2
        a_n = 1.1
        b_n = 2.4
        compute_steady_state_kalman_gain = lambda ti, omega_n_dll, omega_n_pll: block_diag(
            compute_2nd_order_steady_state_kalman_gain(ti, omega_n_dll, xi),
            compute_3rd_order_steady_state_kalman_gain(ti, omega_n_pll, a_n, b_n),
        )
    
    M_cnr_est = kwargs.get('M_cnr_est', 20)
    cnr_smoother_alpha = kwargs.get('cnr_smoother_alpha', .002)

#     print('Nt: {0} \nCorr spacing: {1:3.3f}\tEPL spacing: {2:3.3f}'.format(Nt, chip_delta, epl_chip_spacing))
#     print('Nx: {0:3.3f}'.format(Nx))
    corr = zeros((Nt, len(chip_offsets)), dtype=complex)  # correlator outputs
    
    x_prior = zeros((Nt, Nx))
    x_post = zeros((Nt, Nx))
    P_prior = zeros((Nt, Nx, Nx))
    P_post = zeros((Nt, Nx, Nx))
    z_prior = zeros((Nt, Ny))
    z_post = zeros((Nt, Ny))
    
    cnr_meas = zeros((Nt,))
    cnr_smooth = zeros((Nt,))
    
    x_prior[0, :] = x_post[0, :] = 0
    P_prior[0, :] = P_post[0, :] = diag([.5, 1, 1, 1]) if Nx == 4 else diag([.5, 1, 1, 1, 1])
    
    t_blk = arange(samples_per_block) / samp_rate
    
    x_post[0, 0] = code_phase_acq
    x_post[0, 1] = signal['code_rate'] * (1 + doppler_acq / signal['f_carr'])
    x_post[0, 2] = phi_acq
    x_post[0, 3] = doppler_acq
    
    cnr_meas[0] = cnr_smooth[0] = cnr_acq
    
    time[0] = start_sample / samp_rate
    
    for i in range(1, Nt):
        print('\r {0:07} / {1:07}'.format(i, Nt), end='')
        
        block_sample_index = start_sample + i * samples_per_block
        block = generate_sample_block(f, block_sample_index, samples_per_block)
#         block += numpy.random.randn(len(block)) + 1j * numpy.random.randn(len(block))
        time[i] = block_sample_index / samp_rate
        dt = time[i] - time[i - 1]
        ti = samples_per_block / samp_rate
        
        A = create_A(dt)
        Q = create_Q(dt)
        
        # Propagate state
        x_prior[i] = A @ x_post[i - 1]
        P_prior[i] = A @ P_post[i - 1] @ A.T + Q

        # Extract state
        if carrier_model_order == 2:
            chip0, chip_rate, phi0, phi_rate = x_prior[i]
        elif carrier_model_order == 3:
            chip0, chip_rate, phi0, phi_rate, phi_accel = x_prior[i]
        
        # Correlate
        for j, chip_offset in enumerate(chip_offsets):
            s_ref = signal['sample_function'](t_blk, chip0 + chip_offset, chip_rate, phi0, phi_rate, phi_accel)
            corr[i, j] = numpy.sum(block * conj(s_ref))

        # Estimate state errors
        corr_early, corr_prompt, corr_late = (corr[i, epl_indices[k]] for k in range(3))
        chip_error = epl_chip_spacing * (abs(corr_late) - abs(corr_early)) / (abs(corr_early) + abs(corr_late) + 2 * abs(corr_prompt))
        phase_error = angle(corr_prompt)
#         phase_error = numpy.arctan(corr_prompt.imag / corr_prompt.real)
        z_prior[i, :] = [chip_error, phase_error]
        
        # Estimate C/N0
        im0 = max(1, i - M_cnr_est)
        if i - im0 > 5:
            corr_prompt_samples = corr[im0:i + 1, epl_indices[1]]
            M2 = numpy.mean(abs(corr_prompt_samples)**2)
            M4 = numpy.mean(abs(corr_prompt_samples)**4)
            cnr_meas[i] = sqrt(2 * M2**2 - M4) / (M2 - sqrt(2 * M2**2 - M4)) / ti
            if numpy.isnan(cnr_meas[i]):
                cnr_meas[i] = .1
        else:
            cnr_meas[i] = cnr_meas[i - 1]
        cnr_smooth[i] = cnr_smoother_alpha * cnr_meas[i] + (1 - cnr_smoother_alpha) * cnr_smooth[i - 1]
        
        # Compute measurement matrix and covariance
        H = create_H(ti)
        R = compute_R(ti, cnr_meas[i])
        
        # Compute Kalman gain
        K = P_prior[i] @ H.T @ inv(H @ P_prior[i] @ H.T + R)
        # Use steady-state gain
#         K = compute_steady_state_kalman_gain(block_duration, omega_n_dll, omega_n_pll)

        # Update state and error covariance
        x_post[i] = x_prior[i] + K @ z_prior[i]
        P_post[i] = (identity(Nx) - K @ H) @ P_prior[i] @ (identity(Nx) - K @ H).T + K @ R @ K.T
        
        z_post[i] = z_prior[i] - H @ (x_post[i] - x_prior[i])

    print('')
    return time, x_prior, x_post, P_prior, P_post, z_prior, z_post, cnr_meas, cnr_smooth