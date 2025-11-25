from dEECtheo import dEEC, tot_xsec_sum, dEECimprov, dEECimprovNLO, dEEC_NNLL_resum, dEEC_fixed_order
from data import EEC_merged, EEC_Simulate, EECAlephNew
import numpy as np 
import pandas as pd
from iminuit import Minuit
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os, time
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator, LogLocator

EECdata1 = EEC_merged[(EEC_merged['theta']<0.3) & (EEC_merged['Q']>30.) & (EEC_merged['Q']*EEC_merged['theta']<40)].copy().reset_index(drop=True)
EECdata2 = EEC_Simulate[(EEC_Simulate['theta']<0.3) & (EEC_Simulate['Q']>30.)].copy().reset_index(drop=True)
EECdata3= EECAlephNew[EECAlephNew['theta']<0.3].copy().reset_index(drop=True)

def downsample_group(df, n=30):
    """Return ~n evenly spaced points from df."""
    if len(df) <= n:
        return df  # no need to downsample
    idx = np.linspace(0, len(df) - 1, n, dtype=int)
    return df.iloc[idx]

# Apply to each Q
EECdata2 = (
    EECdata2
    .groupby("Q", group_keys=False)
    .apply(lambda df: downsample_group(df, n=15))
)

EECdata2.reset_index(drop=True, inplace=True)

EECdata = EECdata2

Fit_Counter = 0

def normalize_EEC(EECdata, tot_xsec_sum_func, nloop=3, nf=5, aux=1):
    
    EECdata = EECdata.copy()  # avoid modifying original DataFrame

    # Save raw columns
    EECdata["f raw"] = EECdata["f"] /aux
    EECdata["delta f raw"] = EECdata["delta f"] /aux

    # Normalize f and delta f
    EECdata["f"] = EECdata["f raw"] #/ EECdata["Q"].apply(lambda q: tot_xsec_sum_func(q, nloop, nf))
    EECdata["delta f"] = EECdata["delta f raw"] #/ EECdata["Q"].apply(lambda q: tot_xsec_sum_func(q, nloop, nf))

    # Compute fz and delta fz
    EECdata["fz"] = 2 / np.sin(EECdata["theta"]) * EECdata["f"]
    EECdata["delta fz"] = 2 / np.sin(EECdata["theta"]) * EECdata["delta f"]

    return EECdata

EECdata1 = normalize_EEC(EECdata1, tot_xsec_sum, nloop=3, nf=5)
EECdata2 = normalize_EEC(EECdata2, tot_xsec_sum, nloop=3, nf=5)
EECdata3 = normalize_EEC(EECdata3, tot_xsec_sum, nloop=3, nf=5)

Export_Mode = 0

Export_Filename = "Results_improv_sim.csv"

'''
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
'''

def compute_EECimprov(theta, Q, muOverE, gamma_init, bmax, gq, gg, fq, fg, nloop_log, MU0, Lambda):

    z = (1 - np.cos(theta)) / 2
    
    f = theta * dEECimprov(theta, muOverE/2 * Q, gamma_init, bmax, gq, gg, fq, fg, nloop_log, MU0, Lambda)
    dEECz = 2 / np.sin(theta) * f
    
    fimprvnlo = theta * dEECimprovNLO(theta, muOverE/2 * Q, gamma_init, bmax, gq, nloop_log, MU0)
    dEECzimprvnlo = 2 / np.sin(theta) * fimprvnlo
    
    dEECzNNLL = dEEC_NNLL_resum(z, muOverE/2 * Q, 3, 5)
    fNNLL = dEECzNNLL * np.sin(theta)/2
    
    dEECzNLO = dEEC_fixed_order(z, muOverE/2 * Q, 3, 5, 2)
    fNLO = dEECzNLO * np.sin(theta)/2
    
    return (theta, Q, f, z, dEECz, fNNLL, dEECzNNLL, fNLO, dEECzNLO, fimprvnlo, dEECzimprvnlo)

def cost_EECimprov(muOverE: float, Gammaq: float, Gammag: float, bmax: float,
             gnonpert: float, fq: float, fg: float, norm: float, MU0) -> float:
    """Compute the chi-squared cost function for EEC data."""
    
    global Fit_Counter
    print(Fit_Counter)
    Fit_Counter +=1
    
    gamma_init = np.array([Gammaq, Gammag])
    nloop_log = 1
    
    tasks = [
        (theta, Q, muOverE, gamma_init, bmax, gnonpert, gnonpert, fq, fg, nloop_log, MU0, gnonpert)
        for theta, Q in zip(EECdata['theta'], EECdata['Q'])
    ]

    EEC_rows = pool.starmap(compute_EECimprov, tasks)
    
    print('=========================================================')
    pred_df = pd.DataFrame(EEC_rows, columns=["theta", "Q", "dEEC", "z", "dEECz", "dEECNNLL","dEECzNNLL", "dEECNLO","dEECzNLO", "dEECimprvnlo","dEECzimprvnlo"])
    
    EECdataFit = EECdata.copy()
    # Match predictions to EECdata and compute chi-squared
    EECdataFit['pred'] = norm*pred_df['dEEC'].values
    EECdataFit['predz'] = norm*pred_df['dEECz'].values
    
    EECdataFit['predNNLL'] = norm*pred_df['dEECNNLL'].values
    EECdataFit['predzNNLL'] = norm*pred_df['dEECzNNLL'].values
    
    EECdataFit['predNLO'] = norm*pred_df['dEECNLO'].values
    EECdataFit['predzNLO'] = norm*pred_df['dEECzNLO'].values
    
    EECdataFit['predimprvnlo'] = norm*pred_df['dEECimprvnlo'].values
    EECdataFit['predzimprvnlo'] = norm*pred_df['dEECzimprvnlo'].values
    
    EECdataFit['cost'] = ((EECdataFit['predimprvnlo'] - EECdataFit['f']) / EECdataFit['delta f'])**2

    if(Export_Mode == 1):
        EECdataFit.to_csv(f'Output/{Export_Filename}', index=False)
        return EECdataFit
    
    return EECdataFit['cost'].sum()

def plot_EEC_by_theta(PlotDF, filename="Fit_EEC_Exp.pdf", datalabel="PYTHIA"):

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
    nrows = 2
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
                '''
                if col == ncols - 1 and row < n_per_figure - n_plots:
                    axes[row, col].axis('off')
                    continue
                '''
                Qval, group = groups[fig_idx + ax_idx]
                group = group[group['theta'] > 0]
                ax = axes[row, col]

                # --- Prepare scaled data ---
                theta = group['theta']
                inv_theta = 1 / theta

                y_f = inv_theta * group['f']
                y_err = inv_theta * group['delta f']
                y_pred = inv_theta * group['predimprvnlo']
                y_predimp = inv_theta * group['pred']
                y_predNNLL = inv_theta * group['predNNLL']

                # --- Compute y-limits based on the first two datasets only (ignoring NaNs or <=0 values) ---
                y_ref = np.concatenate([y_f[y_f > 0], y_pred[y_pred > 0]])
                ymin_plot, ymax_plot = np.nanmin(y_ref), np.nanmax(y_ref)

                # Add a reasonable margin in log space (e.g., ±0.3 dex)
                log_margin = 0.3
                ymin_plot /= 10**log_margin
                ymax_plot *= 10**log_margin

                # --- Plot the first two normally ---
                
                ax.plot(theta, y_f, '*', markersize=8, label=datalabel, alpha=1.0)
                
                ax.plot(theta, y_pred, label="Imprv. LLA+NLO",color='red', linestyle='-')
                ax.plot(theta, y_predimp, label="Imprv. LLA",color='red', linestyle=':')

                # --- Truncate divergent NNLL curve before plotting ---
                mask = (y_predNNLL > ymin_plot) & (y_predNNLL < ymax_plot)
                ax.plot(theta[mask], y_predNNLL[mask], linestyle='--', label="Collinear NNLL", color='magenta')
                
                #ax.plot(theta[mask], y_predNLO[mask], linestyle=':', label="Collinear NLO", color='blue')
                
                # Text inside plot instead of title
                ax.text(0.05, 0.57, f"Q = {Qval} GeV", transform=ax.transAxes,
                        fontsize=12, fontweight='bold', ha='left', va='top')
                
                ax.set_yscale("log")
                ax.set_xscale("log")
                ax.set_ylim(ymin_plot, ymax_plot)


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
                    ax.set_xlabel("")
                    ax.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True)
                    #ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=None, numticks=4))

                # Reorder legend handles explicitly
                handles, labels = ax.get_legend_handles_labels()
                # Ensure "Theory Fit" first, "PYTHIA" second
                order = [labels.index(datalabel),labels.index("Imprv. LLA+NLO"),labels.index("Imprv. LLA"), labels.index("Collinear NNLL")]
                ax.legend([handles[i] for i in order], [labels[i] for i in order],
                        frameon=False, loc="best", fontsize=11, handlelength=1.45)
                
                ax.grid(True, which="both", ls="--", alpha=0.5)
                
                ax_idx += 1
                
        fig.supxlabel(r"$\mathrm{d}\Sigma_2^{e^+e^-}/(\theta\mathrm{d}\theta)$ v.s. $\theta$", fontsize=14)
        
        plt.tight_layout(pad=0.5)
        plt.savefig(f"Output/{filename}",
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

def plot_EEC_by_theta_exp(PlotDF, filename="Fit_EEC_Exp.pdf", datalabel="PYTHIA", Note=["","","","","",""]):

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


                Qval, group = groups[fig_idx + ax_idx]
                notefig  = Note[fig_idx + ax_idx]
                group = group[group['theta'] > 0]
                ax = axes[row, col]

                # --- Prepare scaled data ---
                theta = group['theta']
                inv_theta = 1 / theta

                y_f = inv_theta * group['f']
                y_err = inv_theta * group['delta f']
                y_pred = inv_theta * group['predimprvnlo']
                y_predimp = inv_theta * group['pred']
                y_predNNLL = inv_theta * group['predNNLL']
                
                # --- Compute y-limits based on the first two datasets only (ignoring NaNs or <=0 values) ---
                y_ref = np.concatenate([y_f[y_f > 0], y_pred[y_pred > 0]])
                ymin_plot, ymax_plot = np.nanmin(y_ref), np.nanmax(y_ref)

                # Add a reasonable margin in log space (e.g., ±0.3 dex)
                log_margin = 0.5
                ymin_plot /= 10**0.35
                ymax_plot *= 10**0.35

                # --- Plot the first two normally ---
                ax.errorbar(theta, y_f, yerr=y_err, fmt='o', capsize=3, markersize=3,
                            label=datalabel, alpha=0.9)
                ax.plot(theta, y_pred, label="Imprv. LLA+NLO",color='red', linestyle='-')
                ax.plot(theta, y_predimp, label="Imprv. LLA",color='red', linestyle=':')

                # --- Truncate divergent NNLL curve before plotting ---
                mask = (y_predNNLL > ymin_plot) & (y_predNNLL < ymax_plot)
                ax.plot(theta[mask], y_predNNLL[mask], linestyle='--', label="Collinear NNLL", color='magenta')
                
                #ax.plot(theta[mask], y_predNLO[mask], linestyle=':', label="Collinear NLO", color='blue')
                
                # Text inside plot instead of title
                ax.text(0.07, 0.62, f"Q = {Qval} GeV", transform=ax.transAxes,
                        fontsize=11, fontweight='bold', ha='left', va='top')
                ax.text(0.075, 0.53, f"{notefig}", transform=ax.transAxes,
                        fontsize=11, fontweight='bold', ha='left', va='top')
                
                ax.set_yscale("log")
                ax.set_xscale("log")
                ax.set_ylim(ymin_plot, ymax_plot)


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
                    #ax.set_xlabel("")
                else:
                    #ax.set_xlabel(r"$\frac{\mathrm{d}{\Sigma_{2}^{e^+e^-}}}{\theta d\theta}$ v.s. $\theta$")
                    ax.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True)
                    #ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=None, numticks=4))

                # Reorder legend handles explicitly
                handles, labels = ax.get_legend_handles_labels()
                # Ensure "Theory Fit" first, "PYTHIA" second
                order = [labels.index(datalabel),labels.index("Imprv. LLA+NLO"),labels.index("Imprv. LLA"), labels.index("Collinear NNLL")]
                ax.legend([handles[i] for i in order], [labels[i] for i in order],
                        frameon=False, loc="lower left", fontsize=11, handlelength = 1.45)
                
                ax.grid(True, which="both", ls="--", alpha=0.5)
                
                ax_idx += 1
        
        fig.supxlabel(r"$\mathrm{d}\Sigma_2^{e^+e^-}/(\theta\mathrm{d}\theta)$ v.s. $\theta$", fontsize=14)
        plt.tight_layout(pad=0.5)
        plt.savefig(f"Output/{filename}",
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

if __name__ == '__main__':
    
    pool = Pool()

    init_params_improve = {
        "muOverE": 1.0,
        "Gammaq": 0.754,
        "Gammag": 0.824,
        "bmax": 1.5,
        "gnonpert": 3.75,
        "fq": 1,
        "fg": 0,
        "norm": 0.42,
        "MU0": 20,
    }

    
    fixed_params_improve = ["Gammaq", "Gammag",
                    "muOverE",
                    "bmax",
                    "fq","fg",
                    "MU0",
                    #"norm"
                    ] 
    
    time_start = time.time()
    
    EECdata = EECdata2

    Fit_Counter = 0
    #'''
    m = Minuit(cost_EECimprov, **init_params_improve)

    for name in fixed_params_improve:
        m.fixed[name] = True
        
    m.limits['gnonpert'] = (0.1,30)
    m.limits['norm'] = (0.1,5)
    m.limits['muOverE'] = (0.01,10)
    m.limits['bmax'] = (1,3.5)
    m.limits['MU0'] = (20,1000)
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
    Export_Filename = 'Results_improv_Sim.csv'
    EECdata = EECdata2
    #TestDF = cost_EEC(**best_fit_params)
    TestDF = cost_EECimprov(**best_fit_params)
    
    best_fit_params["norm"] = 1.0
    Export_Mode = 1
    Export_Filename = 'Results_improv_Exp.csv'
    EECdata = EECdata1
    TestDF1 = cost_EECimprov(**best_fit_params)
    
    best_fit_params["norm"] = 4/9*0.5/1.15
    Export_Mode = 1
    Export_Filename = 'Results_improv_Exp2.csv'
    EECdata = EECdata3
    TestDF2 = cost_EECimprov(**best_fit_params)
    
    #'''
    
    #'''
    TestDF = pd.read_csv('Output/Results_improv_Sim.csv', header=0)
    plot_EEC_by_theta(TestDF, filename="Fit_EEC_Sim.pdf", datalabel="PYTHIA")
    TestDF1 = pd.read_csv('Output/Results_improv_Exp.csv', header=0)
    TestDF2 = pd.read_csv('Output/Results_improv_Exp2.csv', header=0)
    note = ["TASSO","TASSO","TOPAZ","TOPAZ","ALEPH (Note)","OPAL"]
    plot_EEC_by_theta_exp(pd.concat([TestDF1, TestDF2]), filename="Fit_EEC_Exp_Comb.pdf", datalabel="Experiment", Note=note)
    #'''

