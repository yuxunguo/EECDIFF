from dEECtheo import dEEC, tot_xsec_sum
from data import EEC_merged, EEC_Simulate
import numpy as np 
import pandas as pd
import iminuit as Minuit
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os 

#EECdata = EEC_merged[EEC_merged['z']<0.1]
EECdata = EEC_Simulate[EEC_Simulate['z']<0.1]
Export_Mode = 0

def compute_EEC(theta, Q, EoverQ, gamma_init, bmax, gq, gg, fq, fg, nloop_log):
    z = (1 - np.cos(theta)) / 2
    f = dEEC(theta, EoverQ * Q, gamma_init, bmax, gq, gg, fq, fg, nloop_log)
    dEECz = 2 * theta / np.sin(theta) * f
    return (theta, Q, f, z, dEECz)

def cost_EEC(EoverQ: float, Gammaq: float, Gammag: float, bmax: float,
             gq: float, gg: float, fq: float, fg: float) -> float:
    """Compute the chi-squared cost function for EEC data."""
    gamma_init = np.array([Gammaq, Gammag])
    nloop_log = 1
    
    tasks = [
        (theta, Q, EoverQ, gamma_init, bmax, gq, gg, fq, fg, nloop_log)
        for theta, Q in zip(EECdata['theta'], EECdata['Q'])
    ]

    EEC_rows = pool.starmap(compute_EEC, tasks)
    
    pred_df = pd.DataFrame(EEC_rows, columns=["theta", "Q", "dEEC", "z", "dEECz"])
    
    EECdataFit = EECdata.copy()
    # Match predictions to EECdata and compute chi-squared
    EECdataFit['pred'] = pred_df['dEEC'].values
    EECdataFit['cost'] = ((EECdataFit['pred'] - EECdataFit['f']) / EECdataFit['delta f'])**2
    
    if(Export_Mode == 1):
        return EECdataFit
    
    return EECdataFit['cost'].sum()

def plot_EEC_by_theta(EECdata):
    plt.figure(figsize=(8, 5))

    # Sort by theta for smooth curves
    EECdata = EECdata.sort_values(by='theta')

    # Plot each Q group
    for Qval, group in EECdata.groupby('Q'):
        plt.errorbar(group['theta'], group['f'], yerr=group['delta f'],
                     fmt='o', capsize=3, markersize=4, label=f"Data (Q={Qval})", alpha=0.6)

        plt.plot(group['theta'], group['pred'], label=f"Model (Q={Qval})", lw=2)

    plt.xlabel(r"$\theta$")
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel("EEC")
    plt.title("EEC vs $\\theta$, by $Q$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    pool = Pool()
    init_params = {
        "EoverQ": 0.5,
        "Gammaq": 0.754,
        "Gammag": 0.824,
        "bmax": 1.5,
        "gq": 0.35,
        "gg": 0.35,
        "fq": 1,
        "fg": 0
    }
    Export_Mode = 1
    #print(EECdata)
    TestDF = cost_EEC(**init_params)
    plot_EEC_by_theta(TestDF)
    #print()
    
    ''' 
    fixed_params = ["Gammaq", "Gammag","bmax","gq","gg","fq","fg"] 

    m = Minuit(cost_EEC, **init_params)

    for name in fixed_params:
        m.fixed[name] = True
        
    # Run minimization
    m.migrad()  # Minimize cost
    m.hesse()   # Estimate uncertainties
    ndof = len(EECdata.index) - m.nfit
    with open('Output/Fit_summary.txt', 'w', encoding='utf-8') as f:
        print('The chi squared/d.o.f. is: %.2f / %3d ( = %.2f ).\n' % (m.fval, ndof, m.fval/ndof), file = f)
        print('Below are the final output parameters from iMinuit:', file = f)
        print(*m.values, sep=", ", file = f)
        print(*m.errors, sep=", ", file = f)
        print(m.params, file = f)
    '''
    
    