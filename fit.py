from dEECtheo import dEEC, tot_xsec_sum
from data import EEC_merged, EEC_Simulate
import numpy as np 
import pandas as pd
from iminuit import Minuit
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os, time

EECdata1 = EEC_merged[(EEC_merged['z']<0.2) & (EEC_merged['Q']>30.)].copy().reset_index(drop=True)

EECdata1.loc[:,"f raw"]=EECdata1["f"]/2
EECdata1.loc[:,"delta f raw"]=EECdata1["delta f"]/2
EECdata1["f"] = EECdata1["f raw"] / EECdata1["Q"].apply(lambda q: tot_xsec_sum(q, 3, 5))
EECdata1["delta f"] = EECdata1["delta f raw"] / EECdata1["Q"].apply(lambda q: tot_xsec_sum(q, 3, 5))

EECdata1["fz"] = 2/np.sin(EECdata1["theta"]) * EECdata1["f"]
EECdata1["delta fz"] = 2/np.sin(EECdata1["theta"]) * EECdata1["delta f"]

EECdata2 = EEC_Simulate[EEC_Simulate['theta']<0.3].copy().reset_index(drop=True)

EECdata = EECdata2

Export_Mode = 0

def compute_EEC(theta, Q, muOverE, gamma_init, bmax, gq, gg, fq, fg, nloop_log):
    z = (1 - np.cos(theta)) / 2
    f = theta * dEEC(theta, muOverE/2 * Q, gamma_init, bmax, gq/muOverE**2, gg/muOverE**2, fq, fg, nloop_log)
    dEECz = 2 / np.sin(theta) * f
    return (theta, Q, f, z, dEECz)

def cost_EEC(muOverE: float, Gammaq: float, Gammag: float, bmax: float,
             gq: float, gg: float, fq: float, fg: float, norm: float) -> float:
    """Compute the chi-squared cost function for EEC data."""
    gamma_init = np.array([Gammaq, Gammag])
    nloop_log = 1
    
    tasks = [
        (theta, Q, muOverE, gamma_init, bmax, gq, gg, fq, fg, nloop_log)
        for theta, Q in zip(EECdata['theta'], EECdata['Q'])
    ]

    EEC_rows = pool.starmap(compute_EEC, tasks)
    
    pred_df = pd.DataFrame(EEC_rows, columns=["theta", "Q", "dEEC", "z", "dEECz"])
    
    EECdataFit = EECdata.copy()
    # Match predictions to EECdata and compute chi-squared
    EECdataFit['pred'] = norm*pred_df['dEEC'].values
    EECdataFit['predz'] = norm*pred_df['dEECz'].values
    EECdataFit['cost'] = ((EECdataFit['pred'] - EECdataFit['f']) / EECdataFit['delta f'])**2

    if(Export_Mode == 1):
        EECdataFit.to_csv('Output/Results.csv', index=False)
        return EECdataFit
    
    return EECdataFit['cost'].sum()

def plot_EEC_by_theta(PlotDF):
    # Sort by theta for smooth curves
    PlotDF = PlotDF.sort_values(by='theta')

    # Group by Q values
    groups = list(PlotDF.groupby('Q'))
    n_plots = len(groups)
    ncols = 3
    nrows = 2
    n_per_figure = nrows * ncols

    for fig_idx in range(0, n_plots, n_per_figure):
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 12))
        axes = axes.flatten()  # flatten in case we don’t fill all 6 subplots

        for ax_idx, (Qval, group) in enumerate(groups[fig_idx:fig_idx+n_per_figure]):
            ax = axes[ax_idx]

            ax.errorbar(group['theta'], 1/group['theta']*group['f'], yerr=1/group['theta']*group['delta f'],
                        fmt='o', capsize=3, markersize=4, label="Data", alpha=0.6)
            ax.plot(group['theta']  , 1/group['theta']*group['pred'], label="Model", lw=2)

            ax.set_title(f"Q = {Qval}")
            ax.set_xlabel(r"$theta$")
            ax.set_ylabel("dEEC/dtheta")
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.legend()
            ax.grid(True)

        # Hide any unused subplots
        for j in range(ax_idx + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig("Output/FitBmax_3_Exp.pdf", dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    pool = Pool()
    init_params = {
        "muOverE": 0.346,
        "Gammaq": 0.754,
        "Gammag": 0.824,
        "bmax": 3.0,
        "gq": 0.042,
        "gg": 0.042,
        "fq": 1,
        "fg": 0,
        "norm": 0.6*0.346,
    }

    '''
    Export_Mode = 1
    TestDF = cost_EEC(**init_params)
    plot_EEC_by_theta(TestDF)
    print(TestDF['cost'].sum()/len(TestDF))
    '''
    fixed_params = ["Gammaq", "Gammag","bmax",
                    #"gq",
                    "gg",
                    "fq","fg",
                    #"norm"
                    ] 
    #'''
    time_start = time.time()
    
    m = Minuit(cost_EEC, **init_params)

    for name in fixed_params:
        m.fixed[name] = True
    
    m.limits['gq'] = (0,3)
    m.limits['gg'] = (0,3)
    m.limits['norm'] = (0,5)
    m.limits['muOverE'] = (0.01,10)
    m.limits['bmax'] = (1,3)
    # Run minimization
    m.migrad()  # Minimize cost
    m.hesse()   # Estimate uncertainties
    ndof = len(EECdata.index) - m.nfit
    
    time_end = time.time() -time_start
    
    with open('Output/Fit_summary.txt', 'w', encoding='utf-8') as f:
        print('Total running time: %.1f minutes. Total call of cost function: %3d.\n' % ( time_end/60, m.nfcn), file=f)
        print('The chi squared/d.o.f. is: %.2f / %3d ( = %.2f ).\n' % (m.fval, ndof, m.fval/ndof), file = f)
        print('Below are the final output parameters from iMinuit:', file = f)
        print(*m.values, sep=", ", file = f)
        print(*m.errors, sep=", ", file = f)
        print(m.params, file = f)
        
    best_fit_params = m.values.to_dict()
    
    Export_Mode = 1
    EECdata = EECdata1
    TestDF = cost_EEC(**best_fit_params)
    plot_EEC_by_theta(TestDF)
    #'''
