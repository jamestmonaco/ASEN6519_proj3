from numpy import array, sqrt
from scipy.linalg import block_diag


def compute_2nd_order_alpha_beta_from_xi_omega(ti, xi, omega_n):
    return array([
        [2 * xi * omega_n * ti - 3 / 2 * omega_n**2 * ti**2],
        [omega_n**2 * ti]
    ])

def compute_3rd_order_alpha_beta_gamma_from_a_b_omega(ti, a_n, b_n, omega_n):
    return array([
        [11 / 6 * omega_n**3 * ti**3 - 3 / 2 * a_n * omega_n**2 * ti**2 + b_n * omega_n * ti],
        [-2 * omega_n**3 * ti**2 + a_n * omega_n**2 * ti],
        [omega_n**3 * ti]
    ])

def compute_2nd_order_xi_omega_from_alpha_beta(ti, alpha, beta):
    omega_n = sqrt(beta / ti)
    xi = (alpha + 3 / 2 * omega_n**2 * ti**2) / (2 * omega_n * ti)
    return array([[xi], [omega_n]])

def compute_3rd_order_a_b_omega_from_alpha_beta_gamma(ti, alpha, beta, gamma):
    omega_n = (gamma / ti)**(1 / 3)
    b_n = (beta + 2 * omega_n**3 * ti**2) / (omega_n**2 * ti)
    a_n = (alpha - 11 / 6 * omega_n**3 * ti**3 + 3 / 2 * a_n * omega_n**2 * ti**2) / (omega_n * ti)
    return array([[a_n], [b_n], [omega_n]])

def create_2nd_order_Q(dt, sig2_a, sig2_adot):
    Q = array([
        [dt * sig2_a, dt**2 / 2 * sig2_adot],
        [dt**2 / 2 * sig2_adot, dt * sig2_adot]])
    return Q

def create_3rd_order_Q(dt, sig2_a, sig2_adot, sig2_addot):
    Q = array([
        [dt * sig2_a + dt**3 / 3 * sig2_adot - dt**5 / 20 * sig2_addot, dt**2 / 2 * sig2_adot + dt**4 / 8 * sig2_addot, dt**3 / 6 * sig2_addot],
        [dt**2 / 2 * sig2_adot + dt**4 / 8 * sig2_addot, dt * sig2_adot + dt**3 / 3 * sig2_addot, dt**2 / 2 * sig2_addot],
        [dt**3 / 6 * sig2_addot, dt**2 / 2 * sig2_addot, dt * sig2_addot]
    ])
    return Q

def compute_2nd_order_steady_state_kalman_gain(ti, omega_n, xi):
    K = array([
        [2 * xi * omega_n * ti - 3 / 2 * omega_n**2 * ti**2],
        [omega_n**2 * ti]
    ])
    return K

def compute_3rd_order_steady_state_kalman_gain(ti, omega_n, a_n, b_n):
    K = array([
        [(11 * omega_n**3 * ti**3 - 9 * a_n * omega_n**2 * ti**2 + 6 * b_n * omega_n * ti) / 6],
        [-2 * omega_n**3 * ti**2 + a_n * omega_n**2 * ti],
        [omega_n**3 * ti]
    ])
    return K

def compute_2nd_order_DLL_2nd_order_PLL_Q_matrix(a1, a2, dt, sig2_dG, sig2_ddG, sig2_dI, sig2_dMr, sig2_dMp):
    q11 = a1**2 * (dt * (sig2_dG + sig2_dI + sig2_dMr) + dt**3 / 3 * sig2_ddG)
    q22 = a1**2 * (dt * sig2_ddG)
    q12 = a1**2 * (dt**2 / 2 * sig2_ddG)
    q33 = a2**2 * (dt * (sig2_dG + sig2_dI + sig2_dMp) + dt**3 / 3 * sig2_ddG)
    q44 = a2**2 * (dt * sig2_ddG)
    q34 = a2**2 * (dt**2 / 2 * sig2_ddG)
    q13 = a1 * a2 * (dt * (sig2_dG - sig2_dI) + dt**3 / 3 * sig2_ddG)
    q14 = a1 * a2 * (dt**2 / 2 * sig2_ddG)
    q23 = a1 * a2 * (dt**2 / 2 * sig2_ddG)
    q24 = a1 * a2 * (dt * sig2_ddG)
    Q = array([
        [q11, q12, q13, q14],
        [q12, q22, q23, q24],
        [q13, q23, q33, q34],
        [q14, q24, q34, q44],
    ])
    return Q

def compute_2nd_order_DLL_3rd_order_PLL_Q_matrix(dt, a1, a2, sig2_dG, sig2_ddG, sig2_dddG, sig2_dI, sig2_dMr, sig2_dMp):
    q11 = a1**2 * (dt * (sig2_dG + sig2_dI + sig2_dMr) + dt**3 / 3 * sig2_ddG)
    q22 = a1**2 * (dt * sig2_ddG)
    q12 = a1**2 * (dt**2 / 2 * sig2_ddG)
    q33 = a2**2 * (dt * (sig2_dG + sig2_dI + sig2_dMp) + dt**3 / 3 * sig2_ddG + dt**5 / 20 * sig2_dddG)
    q44 = a2**2 * (dt * sig2_ddG + dt**3 / 3 * sig2_dddG)
    q55 = a2**2 * (dt * sig2_dddG)
    q34 = a2**2 * (dt**2 / 2 * sig2_ddG + dt**4 / 8 * sig2_dddG)
    q35 = a2**2 * (dt**3 / 6 * sig2_dddG)
    q45 = a2**2 * (dt**2 / 2 * sig2_dddG)
    q13 = a1 * a2 * (dt * (sig2_dG - sig2_dI) + dt**3 / 3 * sig2_ddG)
    q14 = a1 * a2 * (dt**2 / 2 * sig2_ddG)
    q15 = a1 * a2 * (0)
    q23 = a1 * a2 * (dt**2 / 2 * sig2_ddG)
    q24 = a1 * a2 * (dt * sig2_ddG)
    q25 = a1 * a2 * (0)
    Q = array([
        [q11, q12, q13, q14, q15],
        [q12, q22, q23, q24, q25],
        [q13, q23, q33, q34, q35],
        [q14, q24, q34, q44, q45],
        [q15, q25, q35, q45, q55],
    ])
    return Q

