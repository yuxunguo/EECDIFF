import numpy as np 
import pandas as pd
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

EEC_Sim1 = pd.read_csv("ee_EEC_data/Simulation/EEC_ee_22GeV.txt", comment="#", index_col=False, header=None, delim_whitespace=True, names = ["z","fz"])
EEC_Sim2 = pd.read_csv("ee_EEC_data/Simulation/EEC_ee_59.5GeV.txt", comment="#", index_col=False, header=None, delim_whitespace=True, names = ["z","fz"])
EEC_Sim3 = pd.read_csv("ee_EEC_data/Simulation/EEC_ee_91.2GeV.txt", comment="#", index_col=False, header=None, delim_whitespace=True, names = ["z","fz"])
EEC_Sim4 = pd.read_csv("ee_EEC_data/Simulation/EEC_ee_300GeV.txt", comment="#", index_col=False, header=None, delim_whitespace=True, names = ["z","fz"])
EEC_Sim5 = pd.read_csv("ee_EEC_data/Simulation/EEC_ee_600GeV.txt", comment="#", index_col=False, header=None, delim_whitespace=True, names = ["z","fz"])

EEC_Sim1['Q']=22.0
EEC_Sim2['Q']=59.5
EEC_Sim3['Q']=91.2
EEC_Sim4['Q']=300.0
EEC_Sim5['Q']=600.0

EEC_Simulate = pd.concat([EEC_Sim1,EEC_Sim2,EEC_Sim3,EEC_Sim4,EEC_Sim5], ignore_index=True)


def invert_z(z):
    # Ensure z is in the valid range [0, 1]
    if np.any((z < 0) | (z > 1)):
        raise ValueError("z must be in the range [0, 1]")
    
    theta = np.arccos(1 - 2 * z)  # in radians
    return theta

EEC_Simulate['theta'] = invert_z(EEC_Simulate['z'])
EEC_Simulate['delta fz'] = 0.1*EEC_Simulate['fz']

EEC_Simulate['f'] = EEC_Simulate['fz'] * np.sin(EEC_Simulate['theta'])/2
EEC_Simulate['delta f'] = 0.1* EEC_Simulate['f']