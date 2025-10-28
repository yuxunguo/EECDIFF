from dEECtheo import dEEC, tot_xsec_sum, dEECimprov
from data import EEC_merged, EEC_Simulate
import numpy as np 
import pandas as pd
from iminuit import Minuit
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os, time
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator, LogLocator

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

def compute_EECimprov(theta, Q, muOverE, gamma_init, bmax, gq, gg, fq, fg, nloop_log, MU0, Lambda):
    z = (1 - np.cos(theta)) / 2
    f = theta * dEECimprov(theta, muOverE/2 * Q, gamma_init, bmax, gq, gg, fq, fg, nloop_log, MU0, Lambda)
    dEECz = 2 / np.sin(theta) * f
    return (theta, Q, f, z, dEECz)

def compute_EEC(theta, Q, muOverE, gamma_init, bmax, gq, gg, fq, fg, nloop_log):
    z = (1 - np.cos(theta)) / 2
    f = theta * dEEC(theta, muOverE/2 * Q, gamma_init, bmax, gq, gg, fq, fg, nloop_log)
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

def cost_EECimprov(muOverE: float, Gammaq: float, Gammag: float, bmax: float,
             gq: float, gg: float, fq: float, fg: float, norm: float, MU0, Lambda) -> float:
    """Compute the chi-squared cost function for EEC data."""
    gamma_init = np.array([Gammaq, Gammag])
    nloop_log = 1
    
    tasks = [
        (theta, Q, muOverE, gamma_init, bmax, gq, gg, fq, fg, nloop_log, MU0, gq)
        for theta, Q in zip(EECdata['theta'], EECdata['Q'])
    ]

    EEC_rows = pool.starmap(compute_EECimprov, tasks)
    
    pred_df = pd.DataFrame(EEC_rows, columns=["theta", "Q", "dEEC", "z", "dEECz"])
    
    EECdataFit = EECdata.copy()
    # Match predictions to EECdata and compute chi-squared
    EECdataFit['pred'] = norm*pred_df['dEEC'].values
    EECdataFit['predz'] = norm*pred_df['dEECz'].values
    EECdataFit['cost'] = ((EECdataFit['pred'] - EECdataFit['f']) / EECdataFit['delta f'])**2

    if(Export_Mode == 1):
        EECdataFit.to_csv('Output/Results_improv.csv', index=False)
        return EECdataFit
    
    return EECdataFit['cost'].sum()

def plot_EEC_by_theta(PlotDF, filename="Fit_EEC_Exp.pdf"):

    # Global font settings
    rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,
        'ytick.labelsize': 11,
        'xtick.labelsize': 11,
        'legend.fontsize': 11,
        'lines.linewidth': 2,
        'lines.markersize': 3,
    })

    os.makedirs("Output", exist_ok=True)

    PlotDF = PlotDF.sort_values(by='theta')
    groups = list(PlotDF.groupby('Q'))
    n_plots = len(groups)

    ncols = 2
    nrows = 3
    n_per_figure = nrows * ncols

    for fig_idx in range(0, n_plots, n_per_figure):
        n_plots_remaining = min(n_per_figure, n_plots - fig_idx)
        nrows_eff = int(np.ceil(n_plots_remaining / ncols))

        # Shared x-axis and tight vertical layout
        fig, axes = plt.subplots(
            nrows_eff, ncols, 
            figsize=(3.0 * ncols, 2.5 * nrows_eff),
            sharex='col',  # share x-axis only within each column
            gridspec_kw={'hspace': 0.0, 'wspace': 0.0}
        )
        axes = np.array(axes).reshape(nrows_eff, ncols)
        
        ax_idx = 0
        for row in range(nrows_eff):
            for col in range(ncols):

                if col == ncols - 1 and row < n_per_figure - n_plots:
                    axes[row, col].axis('off')
                    continue
                
                Qval, group = groups[fig_idx + ax_idx]
                group = group[group['theta'] > 0]
                ax = axes[row, col]

                ax.errorbar(group['theta'], 1/group['theta'] * group['f'],
                            yerr=1/group['theta'] * group['delta f'],
                            fmt='o', capsize=3, markersize=3, label="PYTHIA", alpha=0.6)

                ax.plot(group['theta'], 1/group['theta'] * group['pred'],
                    label="Theory Fit")
                
                # Text inside plot instead of title
                ax.text(0.05, 0.5, f"Q = {Qval} GeV", transform=ax.transAxes,
                        fontsize=12, fontweight='bold', ha='left', va='top')

                ax.set_yscale("log")
                ax.set_xscale("log")

                col_idx = col
                
                if col_idx == 0:  # left-most column
                    ax.yaxis.set_ticks_position('left')
                elif col_idx == ncols - 1:  # right-most column
                    ax.yaxis.set_ticks_position('right')
                else:  # middle columns
                    ax.yaxis.set_ticks_position('none')  # or 'left' if you prefer
            
                # Only hide x-axis labels for top rows, show for bottom row
                if row < nrows_eff - 1:
                    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                    ax.set_xlabel("")
                else:
                    ax.set_xlabel(r"$\frac{dEEC}{\theta d\theta}$ v.s. $\theta$")
                    ax.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True)
                    #ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=None, numticks=4))

                # Reorder legend handles explicitly
                handles, labels = ax.get_legend_handles_labels()
                # Ensure "Theory Fit" first, "PYTHIA" second
                order = [labels.index("PYTHIA"), labels.index("Theory Fit")]
                ax.legend([handles[i] for i in order], [labels[i] for i in order],
                        frameon=False, loc="best", fontsize=12)
                
                ax.grid(True, which="both", ls="--", alpha=0.5)
                
                ax_idx += 1

        plt.tight_layout(pad=0.5)
        plt.savefig(f"Output/{filename}",
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

if __name__ == '__main__':
    pool = Pool()
    init_params = {
        "muOverE": 1.0,
        "Gammaq": 0.754,
        "Gammag": 0.824,
        "bmax": 1.5,
        "gq": 1.0,
        "gg": 1.0,
        "fq": 1,
        "fg": 0,
        "norm": 0.6*0.346,
        #"MU0": 100
    }
    init_params_improve = {
        "muOverE": 1.0,
        "Gammaq": 0.754,
        "Gammag": 0.824,
        "bmax": 1.5,
        "gq": 1.0,
        "gg": 1.0,
        "fq": 1,
        "fg": 0,
        "norm": 0.4,
        "MU0": 20,
        "Lambda": 1.2,
    }

    '''
    Export_Mode = 1
    TestDF = cost_EEC(**init_params)
    plot_EEC_by_theta(TestDF)
    print(TestDF['cost'].sum()/len(TestDF))
    '''
    fixed_params = ["Gammaq", "Gammag",
                    #"muOverE",
                    "bmax",
                    #"gq",
                    #"gg",
                    "fq","fg",
                    #"norm"
                    ] 
    
    fixed_params_improve = ["Gammaq", "Gammag",
                    "muOverE",
                    "bmax",
                    #"gq",
                    #"gg",
                    "fq","fg",
                    #"MU0",
                    "Lambda",
                    #"norm"
                    ] 
    
    #'''
    time_start = time.time()
    '''
    m = Minuit(cost_EEC, **init_params)

    for name in fixed_params:
        m.fixed[name] = True
    '''
    m = Minuit(cost_EECimprov, **init_params_improve)

    for name in fixed_params_improve:
        m.fixed[name] = True
        
    m.limits['gq'] = (0,30)
    m.limits['gg'] = (0,30)
    m.limits['norm'] = (0,5)
    m.limits['muOverE'] = (0.01,10)
    m.limits['bmax'] = (1,3.5)
    m.limits['MU0'] = (20,1000)
    m.limits['Lambda'] = (0,30)
    # Run minimization
    m.migrad()  # Minimize cost
    m.hesse()   # Estimate uncertainties
    ndof = len(EECdata.index) - m.nfit
    
    time_end = time.time() -time_start
    
    with open('Output/Fit_summary_improv.txt', 'w', encoding='utf-8') as f:
        print('Total running time: %.1f minutes. Total call of cost function: %3d.\n' % ( time_end/60, m.nfcn), file=f)
        print('The chi squared/d.o.f. is: %.2f / %3d ( = %.2f ).\n' % (m.fval, ndof, m.fval/ndof), file = f)
        print('Below are the final output parameters from iMinuit:', file = f)
        print(*m.values, sep=", ", file = f)
        print(*m.errors, sep=", ", file = f)
        print(m.params, file = f)
        
    best_fit_params = m.values.to_dict()
    
    Export_Mode = 1
    EECdata = EECdata2
    #TestDF = cost_EEC(**best_fit_params)
    TestDF = cost_EECimprov(**best_fit_params)
    
    plot_EEC_by_theta(TestDF, filename="Fit_EEC_Sim.pdf")
    
    best_fit_params["norm"] = 0.5
    Export_Mode = 1
    EECdata = EECdata1
    #TestDF = cost_EEC(**best_fit_params)
    TestDF = cost_EECimprov(**best_fit_params)
    
    plot_EEC_by_theta(TestDF, filename="Fit_EEC_Exp.pdf")
    #'''
