import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from dEECtheo import dEECimprov
'''
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
'''
data = np.loadtxt("ee_EEC_data/Simulation/EEC_ee_91.2GeV.txt", comments="#")

print(data)
# Split into z and EEC columns
z = data[:, 0]
EEC = data[:, 1]
#'''
# Integrate EEC over z from 0 to 1 using the trapezoidal rule
integral = np.trapz(EEC, z)

print(f"Integral of EEC from z=0 to 1: {integral:.6e}")
#'''