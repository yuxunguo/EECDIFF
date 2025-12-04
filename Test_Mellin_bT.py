import numpy as np
from scipy.special import gamma, kv  # kv is K_v
from scipy.integrate import quad

# Parameters
Gamma0 = 1.0
lambda_i = 2.0
bT_list = np.array([0.1, 0.5, 1.0, 2.0, 5.0])

# Mellin integrand along contour N = c + i t
def mellin_integrand(t, bT, c=1.0):
    N = c + 1j*t
    # f(N) = Gamma0 * lambda^{2-N} * 2^{N-2} * Gamma(N/2)^2
    F_N = Gamma0 * lambda_i**(2-N) * 2**(N-2) * gamma(N/2)**2
    return (bT**(-N) * F_N)  # integrate imaginary part for symmetry

def mellin_inverse(bT, c=1.0, tmax=50):
    # Integrate from 0 to tmax and use symmetry (factor 2)
    result, _ = quad(lambda t: mellin_integrand(t, bT, c) + mellin_integrand(-t, bT, c), 0, tmax, limit=200)
    return result / 2/np.pi  # 1/pi factor due to symmetry

# Exact function
f_exact = Gamma0 * lambda_i**2 * kv(0, lambda_i * bT_list)

# Compute Mellin inverse numerically
f_mellin = np.array([mellin_inverse(bT) for bT in bT_list])

# Print results
for bT, f1, f2 in zip(bT_list, f_exact, f_mellin):
    print(f"bT={bT:.2f}, exact={f1:.6f}, Mellin inverse={f2:.6f}")
