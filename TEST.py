import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from dEECtheo import dEECimprov

Q=100
Gammaq= 0.754
Gammag= 0.824
bmax=1.5
gq= 3.624
gg=1.08
fq=1.0
fg=0.0
nloop_log = 1
mu0=20
Lambda = gq

Gammainit = np.array([Gammaq, Gammag])

def f(theta):
    return theta * dEECimprov(theta, Q, Gammainit, bmax, gq, gg, fq, fg, nloop_log, mu0, Lambda)

result, error = quad(f, 0, np.pi)
print("Integral =", result)
print("Estimated error =", error)