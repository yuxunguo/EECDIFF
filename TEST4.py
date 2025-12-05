import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------------------------------------------------------
# 1) Load the data
# ---------------------------------------------------------
# Put your data into a CSV like:
# Q,qT,gamma_q,gamma_g
# 5.0,0.010000000000000004,0.0450982758916,...

df = pd.read_csv("Output/Gamma_thetaQ_data.csv")

df = df[df['Q'] == 5.0]

qT = df["qT"].values
gamma_q = df["gamma_q"].values
gamma_g = df["gamma_g"].values

# ---------------------------------------------------------
# 2) Example simple fit forms
# ---------------------------------------------------------

# A) Exponential falloff
def fit_exp(qT, A, B, C):
    return A * np.exp(-B*qT) + C

# B) Rational / power-law form
def fit_power(qT, A, B, C):
    return A / (1 + (qT/B) ** C) 

# ---------------------------------------------------------
# 3) Fit gamma_q and gamma_g
# ---------------------------------------------------------

# ---- gamma_q fits ----
popt_q_exp, pcov_q_exp   = curve_fit(fit_exp, qT, gamma_q, p0=[0.05, 0.1, 0.0])
popt_q_pow, pcov_q_pow   = curve_fit(fit_power, qT, gamma_q, p0=[0.05, 0.3, 0.0])

# ---- gamma_g fits ----
popt_g_exp, pcov_g_exp   = curve_fit(fit_exp, qT, gamma_g, p0=[0.05, 0.1, 0.0])
popt_g_pow, pcov_g_pow   = curve_fit(fit_power, qT, gamma_g, p0=[0.05, 0.3, 0.0])

print("\n--- Fit parameters ---")
print("gamma_q (exp):  ", popt_q_exp)
print("gamma_q (power):", popt_q_pow)
print(popt_q_pow[0]/ popt_q_pow[1]**2)
print("gamma_g (exp):  ", popt_g_exp)
print("gamma_g (power):", popt_g_pow)
print(popt_g_pow[0]/ popt_g_pow[1]**2)

# ---------------------------------------------------------
# 4) Plot data vs fits
# ---------------------------------------------------------

qT_plot = np.linspace(min(qT), max(qT), 400)

plt.figure(figsize=(8,6))

# gamma_q
plt.subplot(2,1,1)
plt.scatter(qT, gamma_q, s=12, label="data γ_q", color="black")
plt.plot(qT_plot, fit_exp(qT_plot, *popt_q_exp), label="exp fit", linewidth=2)
plt.plot(qT_plot, fit_power(qT_plot, *popt_q_pow), label="power fit", linewidth=2)
plt.ylabel(r"$\gamma_q$")
plt.yscale("log")
plt.legend()

# gamma_g
plt.subplot(2,1,2)
plt.scatter(qT, gamma_g, s=12, label="data γ_g", color="black")
plt.plot(qT_plot, fit_exp(qT_plot, *popt_g_exp), label="exp fit", linewidth=2)
plt.plot(qT_plot, fit_power(qT_plot, *popt_g_pow), label="power fit", linewidth=2)
plt.ylabel(r"$\gamma_g$")
plt.xlabel(r"$q_T$")
plt.yscale("log")
plt.legend()

plt.tight_layout()
plt.show()
