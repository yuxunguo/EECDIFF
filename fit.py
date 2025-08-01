from dEECtheo import dEEC, tot_xsec_sum
import numpy as np 
import pandas as pd
from iminuit import Minuit
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os 

EECRawdataMAC = pd.read_csv("ee_EEC_data/Raw Data/MAC/Table1.csv", comment="#", header=0, names = ["theta","f","delta f plus","delta f minus"])
EECRawdataMARKII = pd.read_csv("ee_EEC_data/Raw Data/MARKII/Table5.csv", comment="#", header=0, names = ["theta","theta minus", "theta plus","f","delta f stat plus","delta f stat minus", "delta f sys plus","delta f sys minus"])
EECRawdataOPAL = pd.read_csv("ee_EEC_data/Raw Data/OPAL/Table2.csv", comment="#", header=0, names = ["theta","theta minus", "theta plus","f","delta f plus","delta f minus"])
EECRawdataTASSO1 = pd.read_csv("ee_EEC_data/Raw Data/TASSO/Table6.csv", comment="#", header=0, names = ["theta","theta minus", "theta plus","f","delta f plus","delta f minus"])
EECRawdataTASSO2 = pd.read_csv("ee_EEC_data/Raw Data/TASSO/Table7.csv", comment="#", header=0, names = ["theta","theta minus", "theta plus","f","delta f plus","delta f minus"])
EECRawdataTASSO3 = pd.read_csv("ee_EEC_data/Raw Data/TASSO/Table8.csv", comment="#", header=0, names = ["theta","theta minus", "theta plus","f","delta f plus","delta f minus"])
EECRawdataTOPAZ1 = pd.read_csv("ee_EEC_data/Raw Data/TOPAZ/Table1.csv", comment="#", header=0, names = ["theta","theta minus", "theta plus","f","delta f stat plus","delta f stat minus", "delta f sys plus","delta f sys minus"])
EECRawdataTOPAZ2 = pd.read_csv("ee_EEC_data/Raw Data/TOPAZ/Table2.csv", comment="#", header=0, names = ["theta","theta minus", "theta plus","f","delta f stat plus","delta f stat minus", "delta f sys plus","delta f sys minus"])

EECRawdataMAC['f'] = EECRawdataMAC['f']/1000
EECRawdataMAC['delta f plus'] = EECRawdataMAC['delta f plus']/1000
EECRawdataMAC['delta f minus'] = EECRawdataMAC['delta f minus']/1000

EECRawdataTASSO1['f'] = EECRawdataTASSO1['f'] * np.sin(EECRawdataTASSO1['theta']*np.pi/180)
EECRawdataTASSO1['delta f plus'] = EECRawdataTASSO1['delta f plus'] * np.sin(EECRawdataTASSO1['theta']*np.pi/180)
EECRawdataTASSO1['delta f minus'] = EECRawdataTASSO1['delta f minus'] * np.sin(EECRawdataTASSO1['theta']*np.pi/180)

EECRawdataTASSO2['f'] = EECRawdataTASSO2['f'] * np.sin(EECRawdataTASSO2['theta']*np.pi/180)
EECRawdataTASSO2['delta f plus'] = EECRawdataTASSO2['delta f plus'] * np.sin(EECRawdataTASSO2['theta']*np.pi/180)
EECRawdataTASSO2['delta f minus'] = EECRawdataTASSO2['delta f minus'] * np.sin(EECRawdataTASSO2['theta']*np.pi/180)

EECRawdataTASSO3['f'] = EECRawdataTASSO3['f'] * np.sin(EECRawdataTASSO3['theta']*np.pi/180)
EECRawdataTASSO3['delta f plus'] = EECRawdataTASSO3['delta f plus'] * np.sin(EECRawdataTASSO3['theta']*np.pi/180)
EECRawdataTASSO3['delta f minus'] = EECRawdataTASSO3['delta f minus'] * np.sin(EECRawdataTASSO3['theta']*np.pi/180)

EECRawdataTOPAZ1['delta f sys plus'] = EECRawdataTOPAZ1["f"].to_numpy()*np.array(pd.to_numeric(EECRawdataTOPAZ1['delta f sys plus'].str.rstrip('%'), errors='coerce') / 100)
EECRawdataTOPAZ1['delta f sys minus'] = EECRawdataTOPAZ1["f"].to_numpy()*np.array(pd.to_numeric(EECRawdataTOPAZ1['delta f sys minus'].str.rstrip('%'), errors='coerce') / 100)
EECRawdataTOPAZ2['delta f sys plus'] = EECRawdataTOPAZ2["f"].to_numpy()*np.array(pd.to_numeric(EECRawdataTOPAZ2['delta f sys plus'].str.rstrip('%'), errors='coerce') / 100)
EECRawdataTOPAZ2['delta f sys minus'] = EECRawdataTOPAZ2["f"].to_numpy()*np.array(pd.to_numeric(EECRawdataTOPAZ2['delta f sys minus'].str.rstrip('%'), errors='coerce') / 100)

EECRawdataMARKII['delta f plus'] = np.sqrt(EECRawdataMARKII['delta f stat plus'] ** 2 + EECRawdataMARKII['delta f sys plus'] **2  )
EECRawdataMARKII['delta f minus'] = -np.sqrt(EECRawdataMARKII['delta f stat minus'] ** 2 + EECRawdataMARKII['delta f sys minus']**2  )

EECRawdataTOPAZ1['delta f plus'] = np.sqrt(EECRawdataTOPAZ1['delta f stat plus'] ** 2 + EECRawdataTOPAZ1['delta f sys plus'] **2  )
EECRawdataTOPAZ1['delta f minus'] = -np.sqrt(EECRawdataTOPAZ1['delta f stat minus'] ** 2 + EECRawdataTOPAZ1['delta f sys minus']**2  )

EECRawdataTOPAZ2['delta f plus'] = np.sqrt(EECRawdataTOPAZ2['delta f stat plus'] ** 2 + EECRawdataTOPAZ2['delta f sys plus'] **2  )
EECRawdataTOPAZ2['delta f minus'] = -np.sqrt(EECRawdataTOPAZ2['delta f stat minus'] ** 2 + EECRawdataTOPAZ2['delta f sys minus']**2  )

Q_dict = {
    "MAC": 29.0,
    "MARKII": 29.0,
    "OPAL": 91.3,
    "TASSO1": 22.0,
    "TASSO2": 34.8,
    "TASSO3": 43.5,
    "TOPAZ1": 53.3,
    "TOPAZ2": 59.5
}

def extract_core_columns(df, name, Q_map):
    
    df = df.copy().iloc[1:] 
    df['delta f'] = np.maximum(df['delta f plus'].abs(), df['delta f minus'].abs())
    df['Q'] = Q_map.get(name, np.nan)  # Get Q from dict, default NaN if missing
    return df[['theta', 'f', 'delta f', 'Q']].assign(source=name)

frames = []

for (df, name) in [
    (EECRawdataMAC, "MAC"),
    (EECRawdataMARKII, "MARKII"),
    (EECRawdataOPAL, "OPAL"),
    (EECRawdataTASSO1, "TASSO1"),
    (EECRawdataTASSO2, "TASSO2"),
    (EECRawdataTASSO3, "TASSO3"),
    (EECRawdataTOPAZ1, "TOPAZ1"),
    (EECRawdataTOPAZ2, "TOPAZ2"),
]:
    frames.append(extract_core_columns(df, name, Q_dict))

EEC_merged = pd.concat(frames, ignore_index=True)
EEC_merged['theta'] = EEC_merged['theta']/180*np.pi
EEC_merged['z'] = (1-np.cos(EEC_merged['theta'] ))/2

EECdata = EEC_merged[EEC_merged['z']<0.1].copy()

EECdata.loc[:,"f raw"]=EECdata["f"]
EECdata.loc[:,"delta f raw"]=EECdata["delta f"]
EECdata["f"] = EECdata["f raw"] / EECdata["Q"].apply(lambda q: tot_xsec_sum(q, 3, 5))
EECdata["delta f"] = EECdata["delta f raw"] / EECdata["Q"].apply(lambda q: tot_xsec_sum(q, 3, 5))

EECdata["fz"] = 2/np.sin(EECdata["theta"]) * EECdata["f"]
EECdata["delta fz"] = 2/np.sin(EECdata["theta"]) * EECdata["delta f"]

Export_Mode = 0

def compute_EEC(theta, Q, EoverQ, gamma_init, bmax, gq, gg, fq, fg, nloop_log):
    z = (1 - np.cos(theta)) / 2
    f = theta * dEEC(theta, EoverQ * Q, gamma_init, bmax, gq, gg, fq, fg, nloop_log)
    dEECz = 2 / np.sin(theta) * f
    return (theta, Q, f, z, dEECz)

def cost_EEC(EoverQ: float, Gammaq: float, Gammag: float, bmax: float,
             gq: float, gg: float, fq: float, fg: float, norm: float) -> float:
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
    EECdataFit['pred'] = norm*pred_df['dEEC'].values
    EECdataFit['predz'] = norm*pred_df['dEECz'].values
    EECdataFit['cost'] = ((EECdataFit['pred'] - EECdataFit['f']) / EECdataFit['delta f'])**2
    
    if(Export_Mode == 1):
        EECdataFit.to_csv('Output/Results.csv')
        return EECdataFit
    
    return EECdataFit['cost'].sum()

def plot_EEC_by_theta(PlotDF):
    # Sort by theta for smooth curves
    PlotDF = PlotDF.sort_values(by='theta')

    # Group by Q values
    groups = list(PlotDF.groupby('Q'))
    n_plots = len(groups)
    ncols = 4
    nrows = 2
    n_per_figure = nrows * ncols

    for fig_idx in range(0, n_plots, n_per_figure):
        fig, axes = plt.subplots(nrows, ncols, figsize=(24, 12))
        axes = axes.flatten()  # flatten in case we don’t fill all 8 subplots

        for ax_idx, (Qval, group) in enumerate(groups[fig_idx:fig_idx+n_per_figure]):
            ax = axes[ax_idx]

            ax.errorbar(group['z'], group['fz'], yerr=group['delta fz'],
                        fmt='o', capsize=3, markersize=4, label="Data", alpha=0.6)
            ax.plot(group['z'], group['predz'], label="Model", lw=2)

            ax.set_title(f"Q = {Qval}")
            ax.set_xlabel(r"$z$")
            ax.set_ylabel("dEEC/dz")
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.legend()
            ax.grid(True)

        # Hide any unused subplots
        for j in range(ax_idx + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig("Output/myplot.pdf", dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    pool = Pool()
    init_params = {
        "EoverQ": 0.12,
        "Gammaq": 0.754,
        "Gammag": 0.824,
        "bmax": 1.5,
        "gq": 0.35,
        "gg": 0.35,
        "fq": 1,
        "fg": 0,
        "norm": 1.35,
    }
    Export_Mode = 1
    TestDF = cost_EEC(**init_params)
    plot_EEC_by_theta(TestDF)

    fixed_params = ["Gammaq", "Gammag","bmax",
                    #"gq","gg",
                    "fq","fg",
                    #"norm"
                    ] 
    '''
    m = Minuit(cost_EEC, **init_params)

    for name in fixed_params:
        m.fixed[name] = True
    
    m.limits['gq'] = (0,3)
    m.limits['gg'] = (0,3)
    m.limits['norm'] = (0,5)
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
        
    best_fit_params = m.values.to_dict()
    
    Export_Mode = 1

    TestDF = cost_EEC(**best_fit_params)
    plot_EEC_by_theta(TestDF)
    '''
