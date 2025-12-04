import numpy as np
from scipy.special import gamma  # gamma function
from scipy.integrate import quad

# Parameters
Gamma0 = 1.0
lambda_i = 2.0
qT_list = np.array([0.1, 0.5, 1.0, 2.0, 5.0])

# Mellin integrand along vertical contour N = c + i t
def mellin_integrand(t, qT, c=1.0):
    N = c + 1j * t
    F_N = (Gamma0 / 2) * lambda_i**N * gamma(N/2) * gamma(1 - N/2)
    return (qT**(-N) * F_N)  # imaginary part for symmetry

def mellin_inverse(qT, c=1.0, tmax=50):
    # Integrate from 0 to tmax and use symmetry (factor 2)
    result, _ = quad(lambda t: mellin_integrand(t, qT, c) + mellin_integrand(-t, qT, c), 0, tmax, limit=200)
    return result / 2/np.pi  # 1/pi factor due to symmetry

# Exact function
f_exact = Gamma0 / (1 + (qT_list / lambda_i)**2)

# Compute Mellin inverse numerically
f_mellin = np.array([mellin_inverse(qT) for qT in qT_list])

# Print results
for qT, f1, f2 in zip(qT_list, f_exact, f_mellin):
    print(f"qT={qT:.2f}, exact={f1:.6f}, Mellin inverse={f2:.6f}")
