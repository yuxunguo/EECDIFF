import pandas as pd
import matplotlib.pyplot as plt

Int_EEC_LO = pd.read_csv(
    "EEC_in_DIS/DataFile/int0",
    names=["Q", "EEC"],
    engine="python",
    sep=r"[{},\s]+",
    usecols=[1, 2]
)

Int_EEC_NLO = pd.read_csv(
    "EEC_in_DIS/DataFile/int1",
    names=["Q", "EEC"],
    engine="python",
    sep=r"[{},\s]+",
    usecols=[1, 2]
)

Unint_EEC_LO = pd.read_csv(
    "EEC_in_DIS/DataFile/zeta_LO",
    names=["zeta", "dEEC"],
    engine="python",
    sep=r"[{},\s]+",
    usecols=[1, 2]
)

Unint_EEC_LLA = pd.read_csv(
    "EEC_in_DIS/DataFile/zeta_LLA",
    names=["zeta", "dEEC"],
    engine="python",
    sep=r"[{},\s]+",
    usecols=[1, 2]
)

Unint_EEC_LLANLO = pd.read_csv(
    "EEC_in_DIS/DataFile/zeta_LLANLO",
    names=["zeta", "dEEC"],
    engine="python",
    sep=r"[{},\s]+",
    usecols=[1, 2]
)

Unint_EEC_Asym = pd.read_csv(
    "EEC_in_DIS/DataFile/zeta_Asym",
    names=["zeta", "dEEC"],
    engine="python",
    sep=r"[{},\s]+",
    usecols=[1, 2]
)


plt.figure(figsize=(5.85, 3)) 

plt.plot(Int_EEC_LO['Q'], Int_EEC_LO['EEC'],color='magenta',linestyle='--',label = r"LO Theory")
plt.plot(Int_EEC_NLO['Q'], Int_EEC_NLO['EEC'],color='green',label = r"NLO Theory")

#plt.xlim(12.5, 580) 
plt.ylim(0.68, 1.0) 

plt.title(r"Theory predictions of integrated EEC $\Sigma_{2}^{\mathrm{DIS,T}}$ in DIS", fontsize = 14)
plt.xlabel("Q (GeV)", fontsize = 12)
plt.ylabel("$\Sigma_2^{\mathrm{DIS,T}}(Q)$", fontsize = 12)
#plt.xscale("log")
#plt.yscale("log")
plt.xticks([10, 20, 30, 40, 50], ["10", "20", "30", "40", "50"])
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout(pad=0.1)
plt.savefig("Output_DIS/IntegratedEEC_DIS.pdf", format="pdf") 
plt.close()

plt.figure(figsize=(5.5, 3.5)) 

plt.plot(Unint_EEC_LO['zeta'], Unint_EEC_LO['dEEC'],color='blue',linestyle='-',label = r"Full LO")
plt.plot(Unint_EEC_Asym['zeta'], Unint_EEC_Asym['dEEC'],color='blue',linestyle='--',label = r"Asymp. LO")

plt.plot(Unint_EEC_LLA['zeta'], Unint_EEC_LLA['dEEC'],color='red',linestyle='-',label = r"Impr. LLA")
plt.plot(Unint_EEC_LLANLO['zeta'], Unint_EEC_LLANLO['dEEC'],color='red',linestyle='--',label = r"Impr. LLA+NLO")


plt.xlim(0.01, 0.5) 
plt.ylim(0.08, 12) 

plt.title(r"Angular distributions of unintegrated EEC $d\Sigma_{2}^{\mathrm{DIS,T}}/d\zeta$ in DIS", fontsize = 12)
plt.xlabel(r"$\zeta$", fontsize = 12)
plt.ylabel("$d\Sigma_2^{\mathrm{DIS,T}}/d\zeta$", fontsize = 12)
plt.xscale("log")
plt.yscale("log")
plt.xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
           ["0.01", "0.02", "0.05", "0.1", "0.2", "0.5"])
plt.yticks([0.1, 0.2, 0.5, 1, 2, 5, 10],
           ["0.1", "0.2", "0.5", "1", "2", "5", "10"])
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout(pad=0.1)
plt.savefig("Output_DIS/UnintegratedEEC_DIS.pdf", format="pdf") 
plt.close()