from math import gamma, tau
import numpy as np
from scipy.special import psi, j0, zeta
from scipy.integrate import simps
from typing import Tuple, Union
import rundec
import pandas as pd
from iminuit import Minuit
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.text import Text
from matplotlib.patches import Patch,Rectangle
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from functools import cache, lru_cache
from scipy.integrate import quad, quad_vec
from multiprocessing import Pool
import matplotlib.ticker as ticker
from itertools import product

from dEECtheo import Gamma_tilde_Perturbative_Evo, evolop

BMAX_INTEGRAL = 10.0

NF =5 
P =1
NLOOP_ALPHA_S = 3 

def GammaqTLLA(theta: float, Q: float, Gamma_Init: np.array, bmax: float, gq: float, gg: float, nlooplog: int):
    
    def integrand(bT):
        bstar = bT / np.sqrt(1 + bT**2 / bmax**2)
        Gamma_Pert = Gamma_tilde_Perturbative_Evo(Gamma_Init, Q, bstar, nlooplog)
        Gamma_NonP = np.exp(-np.array([gq, gg]) * bT)
        Gamma = Gamma_Pert * Gamma_NonP
        gamma_dot = Gamma
        return bT * j0(theta * bT * Q) * gamma_dot

    integral, _ = quad_vec(integrand, 0, BMAX_INTEGRAL, epsabs=1e-6, epsrel=1e-6, limit=200)
    
    return integral

gammaqT0q = 0.03782223
gammaqT0g = 0.03090976
LambdaqT = 3.4004869

def gammqT_ref(qT):
    resultq = gammaqT0q/(1+(qT/LambdaqT)**2)
    resultg = gammaqT0g/(1+(qT/LambdaqT)**2)
    return np.array([resultq, resultg])

def gammaqT_Mellin(s):
    
    resultq = -np.pi/2 * gammaqT0q * LambdaqT ** (-s) /np.sin(np.pi*s/2)
    resultg = -np.pi/2 * gammaqT0g * LambdaqT ** (-s) /np.sin(np.pi*s/2)
    
    return np.array([resultq, resultg])

def GammaqTMellin_Ref(qT, mu0, mu, Gammainit, fixed_j):
    
    # Mellin integrand along contour s = c + i t
    def mellin_integrand(t, qT, c):
        s = c + 1j*t
        Fs = Gammainit(s)
        evos = evolop(fixed_j, NF, P, mu, mu0, NLOOP_ALPHA_S)
        Fs= Fs @ evos
        return (qT**(s) * Fs)  # integrate imaginary part for symmetry
    
    c= -0.05
    tmax = 200
    result, _ = quad_vec(lambda t: mellin_integrand(t, qT, c) + mellin_integrand(-t, qT, c), 0, tmax, limit=200)
    return np.real(result) / 2/np.pi  # 1/pi factor due to symmetry

def GammaqTMellin(qT, mu0, mu, Gammainit):
    
    # Mellin integrand along contour s = c + i t
    def mellin_integrand(t, qT, c):
        s = c + 1j*t
        Fs = Gammainit(s)
        evos = evolop(s-1+5,NF, P, mu, mu0, NLOOP_ALPHA_S)
        Fs= Fs @ evos
        return (qT**(s) * Fs)  # integrate imaginary part for symmetry
    
    c= -0.05
    tmax = 200
    result, _ = quad_vec(lambda t: mellin_integrand(t, qT, c) + mellin_integrand(-t, qT, c), 0, tmax, limit=200)
    return np.real(result) / 2/np.pi  # 1/pi factor due to symmetry

def Unintegrated_EEC_Jet_Scaling_Plt(qTsmall, qTlarge, Qlst):
    
    MU0 = 20.0
    def compute_gamma_curve(qT):

        Gq_vals = []
        Gg_vals = []
        
        for Q in Qlst:
            #Iq, Ig = GammaImprov(th, qT/th, Gammainit, bmax, Gnonpert, nlooplog, MU0)
            Iq, Ig = GammaqTMellin(qT, MU0, Q, gammaqT_Mellin)
            Gq_vals.append(Iq)
            Gg_vals.append(Ig)

        return np.array(Qlst), np.array(Gq_vals), np.array(Gg_vals)

    #fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=False)
    
    qTlarge_Ref = 10*qTlarge
    
    qTsmall_Ref = 0.1*qTsmall
    
    ref3qlst = []
    ref5qlst = []
    for Q in Qlst:
        J = 2
        
        #ref3q = np.array([1,0]) @ evolop(J, NF, P, Q, MU0, NLOOP_ALPHA_S) @ np.array([1,1])
        #ref5q = np.array([1,0]) @ evolop(J+2, NF, P, Q, MU0, NLOOP_ALPHA_S) @ np.array([1,1])
        ref3q, ref3g = GammaqTMellin_Ref(qTlarge_Ref, MU0, Q, gammaqT_Mellin, J)
        ref5q, ref5g = GammaqTMellin_Ref(qTsmall_Ref, MU0, Q, gammaqT_Mellin, J+2)
        ref3qlst.append(ref3q)
        ref5qlst.append(ref5q)
    
    
    Qvals, GqqTsmall, GgqTsmall = compute_gamma_curve(qTsmall)
    Qvals, GqqTlarge, GgqTlarge = compute_gamma_curve(qTlarge)
    
    plt.figure(figsize=(5.25,3.75))

    # Calculated curves with markers
    plt.plot(Qvals, GqqTsmall/GqqTsmall[0],
            label=fr"$\Gamma_q^{{\rm{{Imprv.}}}}$ ($\mu$,$q_T$ = {qTsmall} GeV)",
            linestyle='', marker='o', markersize=6, 
            markerfacecolor='none', markeredgecolor='red')
    plt.plot(Qvals, GqqTlarge/GqqTlarge[0],
            label=fr"$\Gamma_q^{{\rm{{Imprv.}}}}$ ($\mu$,$q_T$ = {qTlarge} GeV)",
            linestyle='', marker='*', markersize=8,
            markerfacecolor='none', markeredgecolor='magenta')
    
    # Reference curves
    plt.plot(Qvals, np.array(ref3qlst)/ref3qlst[0],
            label=fr"Reference: $\gamma^3$-scaling", linestyle='-', color = 'red')
    plt.plot(Qvals, np.array(ref5qlst)/ref5qlst[0],
            label=fr"Reference: $\gamma^5$-scaling", linestyle='--', color = 'magenta')

    plt.xscale("log")
    #plt.yscale("log")
    #plt.ylim(0.63, 1.01) 
    plt.xlabel(r"$\mu$ [GeV]")
    plt.ylabel(fr"$\Gamma_q^{{\mathrm{{imprv.}}}}(\mu,q_T)$")
    plt.title(fr"$\mu$-scaling of improved EEC jet functions $\Gamma_q^{{\mathrm{{imprv.}}}}(\mu,q_T)$")
    plt.legend(markerscale=1,handlelength=1.35)
    plt.tight_layout()
    plt.grid(True, alpha=0.5)
    plt.savefig("Output_Mellin/Gamma_muscale.pdf", bbox_inches='tight')

def Unintegrated_EEC_Jet_Table(qT, Qlst):

    MU0 = 20.0
    def compute_gamma_curve(Q):

        Gq_vals = []
        Gg_vals = []
        for qT_val in qT:
            Iq, Ig = GammaqTMellin(qT_val, MU0, Q, gammaqT_Mellin)
            Gq_vals.append(Iq)
            Gg_vals.append(Ig)

        return qT, np.array(Gq_vals), np.array(Gg_vals)

    # --------------------------
    # Collect all data here
    # --------------------------
    rows = []

    for Q in Qlst:
        thetaQ, Gq, Gg = compute_gamma_curve(Q)
        for qT_val, gq_val, gg_val in zip(qT, Gq, Gg):
            rows.append({
                "Q": Q,
                "qT": qT_val,
                "gamma_q": gq_val,
                "gamma_g": gg_val
            })

    df = pd.DataFrame(rows)

    # Save to CSV
    df.to_csv("Output_Mellin/Gamma_thetaQ_data_Comp.csv", index=False)

def Unintegrated_EEC_Jet_Plot():
    
    df = pd.read_csv("Output_Mellin/Gamma_thetaQ_data_Comp.csv")
    
    df2 = pd.read_csv("Output/Gamma_thetaQ_data_Comp.csv")
    # Get unique Q values
    Qlst = df['Q'].unique()

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=False)

    # --- Quark ---
    for Q in Qlst:
        subset = df[df['Q'] == Q]
        ax[0].plot(subset['qT'], subset['gamma_q'], label=fr"$\bar E$ = {Q:.0f} GeV", linestyle='-')
        subset2 = df2[df2['Q'] == Q]
        ax[0].plot(subset2['qT'], subset2['gamma_q'], linestyle='--')
    
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$\theta \bar E = q_T$")
    ax[0].set_ylabel(r"$\Gamma_q(\theta \bar E)$")
    ax[0].set_title("Quark Contribution")
    ax[0].legend()

    # --- Gluon ---
    for Q in Qlst:
        subset = df[df['Q'] == Q]
        ax[1].plot(subset['qT'], subset['gamma_g'], label=fr"$\bar E$ = {Q:.0f} GeV", linestyle='-')
        subset2 = df2[df2['Q'] == Q]
        ax[1].plot(subset2['qT'], subset2['gamma_g'], linestyle='--')

    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$\theta \bar E = q_T$")
    ax[1].set_ylabel(r"$\Gamma_g(\theta \bar{E})$")
    ax[1].set_title("Gluon Contribution")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig("Output_Mellin/Gamma_thetaQ_New.pdf", bbox_inches='tight')

if __name__ == '__main__':
    '''
    Qlst = np.exp(np.linspace(np.log(50), np.log(1000), 15))

    Unintegrated_EEC_Jet_Scaling_Plt(1.0, 20.0, Qlst)
    '''
    
    Qlst= np.array([20.])
    
    qT = np.exp(np.linspace(np.log(10**(-2)), np.log(20), 50))
    
    Unintegrated_EEC_Jet_Table(qT, Qlst)
    
    Unintegrated_EEC_Jet_Plot()