from dEECtheo import dEEC, tot_xsec_sum
import numpy as np 
import pandas as pd
import iminuit as Minuit
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

EECdata = EEC_merged[EEC_merged['z']<0.1]

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
    
    