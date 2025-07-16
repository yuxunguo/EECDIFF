import numpy as np
from scipy.special import psi, j0
from scipy.integrate import simps
from typing import Tuple, Union
import rundec
import pandas as pd
import matplotlib.pyplot as plt


NC = 3
CF = (NC**2 - 1) / (2 * NC)
CA = NC
CG = CF - CA/2
TF = 0.5

B00 = 11./3. * CA
B01 = -4./3. * TF
B10 = 34./3. * CA**2
B11 = -20./3. * CA*TF - 4. * CF*TF

crd = rundec.CRunDec()

def AlphaS(nloop: int, nf: int, mu: float):
    
    MZ = 91.1876
    AlphaS_MZ = 0.1185
    
    return crd.AlphasExact(AlphaS_MZ, MZ, mu, nf, nloop)

def beta0(nf: int) -> float:
    """ LO beta function of pQCD, will be used for LO GPD evolution. """
    return - B00 - B01 * nf

def S1(z: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
    """Harmonic sum S_1."""
    return np.euler_gamma + psi(z+1)

def singlet_LO(n: Union[complex, np.ndarray], nf: int, p: int, prty: int = 1) -> np.ndarray:
    """Singlet LO anomalous dimensions.

    Args:
        n (complex): which moment (= Mellin moment for integer n)
        nf (int): number of active quark flavors
        p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et)
        prty (int): C parity, irrelevant at LO

    Returns:
        2x2 complex matrix ((QQ, QG),
                            (GQ, GG))

    This will work as long as n, nf, and p can be broadcasted together.

    """

    epsilon = 0.00001 * ( n == 1)

    # Here, I am making the assumption that a is either 1 or -1
    qq0 = np.where(p>0,  CF*(-3.0-2.0/(n*(1.0+n))+4.0*S1(n)),           CF*(-3.0-2.0/(n*(1.0+n))+4.0*S1(n)))
    qg0 = np.where(p>0,  (-4.0*nf*TF*(2.0+n+n*n))/(n*(1.0+n)*(2.0+n)),  (-4.0*nf*TF*(-1.0+n))/(n*(1.0+n)) )
    gq0 = np.where(p>0,  (-2.0*CF*(2.0+n+n*n))/((-1.0+n + epsilon)*n*(1.0+n)),    (-2.0*CF*(2.0+n))/(n*(1.0+n)))
    gg0 = np.where(p>0,  -4.0*CA*(1/((-1.0+n + epsilon)*n)+1/((1.0+n)*(2.0+n))-S1(n)) -11*CA/3. + 4*nf*TF/3., \
        -4.0*CA*(2/(n*(1.0+n))-S1(n)) -11*CA/3. + 4*nf*TF/3. )

    # all of the four above have shape (N)
    # more generally, if p is a multi dimensional array, like (N1, N1, N2)... Then this could also work

    qq0_qg0 = np.stack((qq0, qg0), axis=-1)
    gq0_gg0 = np.stack((gq0, gg0), axis=-1)

    return np.stack((qq0_qg0, gq0_gg0), axis=-2)# (N, 2, 2)

def lambdaf(n: complex, nf: int, p: int, prty: int = 1) -> np.ndarray:
    """Eigenvalues of the LO singlet anomalous dimensions matrix.

    Args:
        n (complex): which moment (= Mellin moment for integer n)
        nf (int): number of active quark flavors
        p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et)
        prty (int): 1 for NS^{+}, -1 for NS^{-}, irrelevant at LO

    Returns:
        lam[a, k]
        a in [+, -] and k is MB contour point index

    Normally, n and nf should be scalars. p should be (N)
    More generally, as long as they can be broadcasted, any shape is OK.

    """
    # To avoid crossing of the square root cut on the
    # negative real axis we use trick by Dieter Mueller
    gam0 = singlet_LO(n, nf, p, prty) # (N, 2, 2)
    aux = ((gam0[..., 0, 0] - gam0[..., 1, 1]) *
           np.sqrt(1. + 4.0 * gam0[..., 0, 1] * gam0[..., 1, 0] /
                   (gam0[..., 0, 0] - gam0[..., 1, 1])**2)) # (N)
    lam1 = 0.5 * (gam0[..., 0, 0] + gam0[..., 1, 1] - aux) # (N)
    lam2 = lam1 + aux  # (N)
    return np.stack([lam1, lam2], axis=-1) # shape (N, 2)

def projectors(n: complex, nf: int, p: int, prty: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Projectors on evolution quark-gluon singlet eigenaxes.

    Args:
        n (complex): which moment (= Mellin moment for integer n)
        nf (int): number of active quark flavors
        p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et)
        prty (int): 1 for NS^{+}, -1 for NS^{-}, irrelevant at LO

    Returns:
         lam: eigenvalues of LO an. dimm matrix lam[a, k]  # Eq. (123)
          pr: Projector pr[k, a, i, j]  # Eq. (122)
               k is MB contour point index
               a in [+, -]
               i,j in {Q, G}

    n and nf will be scalars
    p will be shape (N)
    prty should be scalar (but maybe I can make it work with shape N)

    """
    gam0 = singlet_LO(n, nf, p, prty)    # (N, 2, 2)
    lam = lambdaf(n, nf, p, prty)        # (N, 2)
    den = 1. / (lam[..., 0] - lam[..., 1]) #(N)
    # P+ and P-
    ssm = gam0 - np.einsum('...,ij->...ij', lam[..., 1], np.identity(2)) #(N, 2, 2)
    ssp = gam0 - np.einsum('...,ij->...ij', lam[..., 0], np.identity(2)) #(N, 2, 2)
    prp = np.einsum('...,...ij->...ij', den, ssm) # (N, 2, 2)
    prm = np.einsum('...,...ij->...ij', -den, ssp) # (N, 2, 2)
    # We insert a-axis before i,j-axes, i.e. on -3rd place
    pr = np.stack([prp, prm], axis=-3) # (N, 2, 2, 2)
    return lam, pr # (N, 2) and (N, 2, 2, 2)

def evolop(j: complex, nf: int, p: int, mu: float, mu_init: float, nloop_alphaS: int):
    """Leading order GPD evolution operator E(j, nf, mu)[a,b].

    Args:
         j: MB contour points (Note: n = j + 1 !!)
         nf: number of effective fermion
         p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et)
         mu: final scale of evolution 

    Returns:
         Evolution operator E(j, nf, mu)[a,b] at given j nf and mu as 3-by-3 matrix
         - a and b are in the flavor space (non-singlet, singlet, gluon)

    In original evolop function, j, nf, p, and mu are all scalars.
    Here, j and nf will be scalars.
    p and mu will have shape (N)

    """
    #Alpha-strong ratio.
    R = AlphaS(nloop_alphaS, nf, mu)/AlphaS(nloop_alphaS, nf, mu_init) # shape N
    R = np.array(R)

    #LO singlet anomalous dimensions and projectors
    lam, pr = projectors(j+1, nf, p)    # (N, 2) (N, 2, 2, 2)

    #LO pQCD beta function of GPD evolution
    b0 = beta0(nf) # scalar

    #Singlet LO evolution factor (alpha(mu)/alpha(mu0))^(-gamma/beta0) in (+,-) space
    Rfact = R[..., np.newaxis]**(-lam/b0) # (N, 2)     

    #Singlet LO evolution matrix in (u+d, g) space
    """
    # The Gepard code by K. Kumericki reads:
    evola0ab = np.einsum('kaij,ab->kabij', pr,  np.identity(2))
    evola0 = np.einsum('kabij,bk->kij', evola0ab, Rfact)
    # We use instead
    """ 
    evola0 = np.einsum('...aij,...a->...ij', pr, Rfact) # (N, 2, 2)

    return evola0 # (N) and (N, 2, 2)


J = 3 - 1
NF = 5
P = 1
nloop = 3
mu0 = 2
    
    
def Gamma_Evo(Gamma_Init: np.array, mu: float):
    
    Evo0 = evolop(J, NF, P, mu, mu0, nloop)
    Gamma_mu = 1 - np.einsum('i,ij->j', 1-Gamma_Init, Evo0)
    
    return Gamma_mu

def Gamma_tilde_Perturbative_Evo(Gamma_Init: np.array, mu:float, bT: float):
    
    mub = 2 * np.exp(-np.euler_gamma) /bT
    
    Gamma_mub = Gamma_Evo(Gamma_Init, mub)

    Evo = evolop(J, NF, P, mu, mub, nloop)

    return np.einsum('i,ij->j', Gamma_mub, Evo)

def Gamma_tilde_Resum_Evo(Gamma_Init: np.array, mu:float, bT: float, bmax: float, gq: float, gg: float):
    
    bstar = bT / np.sqrt( 1 + bT**2/ (bmax**2) )
    
    Gamma_Pert = Gamma_tilde_Perturbative_Evo(Gamma_Init, mu, bstar)
    
    Gamma_NonP = np.exp(-np.array([gq,gg]) * bT**2)
        
    return Gamma_Pert*Gamma_NonP

def dEEC(theta: float, Q: float, Gamma_Init: np.array, bmax: float, gq: float, gg: float, fq: float, fg:float):
    
    fqg=np.array([fq,fg])
    epsilon = 10. ** (-8)
    bTlst = np.linspace(epsilon,10,2001)
    
    gammalst = [np.dot(Gamma_tilde_Resum_Evo(Gamma_Init, Q, bT, bmax, gq, gg), fqg) for bT in bTlst]
    
    besselJ0lst = j0(theta*bTlst*Q)
    
    ylst = bTlst * besselJ0lst * gammalst
    
    integral = simps(ylst, bTlst)   
    
    '''
    plt.plot(bTlst, bTlst*gammalst, marker='o',label = f"Q=20 GeV")
    
    plt.xlabel("bT (GeV$^{-1}$)")
    plt.ylabel("dEEC")
    plt.title("dEEC vs bT")
    plt.grid(True)
    plt.legend()
    plt.yscale("log")
    plt.show()
    '''
    return integral

def Gamma_cal_plt(gammainit, Qlst):
    
    GammaEvolst=np.array([Gamma_Evo(gammainit, Q)   for Q in Qlst])
    
    GammaEvolstq = GammaEvolst[:,0]
    GammaEvolstg = GammaEvolst[:,1]
    
    plt.plot(Qlst, GammaEvolstq,color='red',marker="o",label = r"$\Gamma_q$")
    plt.plot(Qlst, GammaEvolstg,color='blue',marker="s",label = r"$\Gamma_g$")
    plt.plot(Qlst, 1-GammaEvolstq,linestyle='--',color='red',marker="o", markerfacecolor='none',label = r"$\Gamma'_q$")
    plt.plot(Qlst, 1-GammaEvolstg,color='blue',linestyle='--',marker="s", markerfacecolor='none',label = r"$\Gamma'_g$")
    
    plt.xlabel("Q (GeV)")
    plt.ylabel("$\Gamma_i$")
    plt.xscale("log")
    plt.grid(True)
    plt.legend()
    plt.show()

def Gamma_tilde_cal_plt(gammainit, Qlst, bTlst):
    
    Gamma_tilde = []
    for bT in bTlst:
        for Q in Qlst:
            f = Gamma_tilde_Perturbative_Evo(gammainit, Q, bT)
            Gamma_tilde.append((bT, Q, f[0],f[1]))
            
    df = pd.DataFrame(Gamma_tilde, columns=["bT", "Q", "Gammatq", "Gammatg"])
    
    df_Q1 = df[df["Q"] == Qlst[0]]
    df_Q2 = df[df["Q"] == Qlst[1]]
    df_Q3 = df[df["Q"] == Qlst[2]]
    df_Q4 = df[df["Q"] == Qlst[3]]
    df_Q5 = df[df["Q"] == Qlst[4]]
    
    plt.plot(df_Q1["bT"], df_Q1["Gammatq"], marker='o',color='red',label = f"Q={Qlst[0]} GeV")
    plt.plot(df_Q2["bT"], df_Q2["Gammatq"], marker='s',color='red',label = f"Q={Qlst[1]} GeV")
    plt.plot(df_Q3["bT"], df_Q3["Gammatq"], marker='^',color='red',label = f"Q={Qlst[2]} GeV")
    plt.plot(df_Q4["bT"], df_Q4["Gammatq"], marker='*',color='red',label = f"Q={Qlst[3]} GeV")
    plt.plot(df_Q5["bT"], df_Q5["Gammatq"], marker='+',color='red',label = f"Q={Qlst[4]} GeV")
    
    
    plt.plot(df_Q1["bT"], df_Q1["Gammatg"], marker='o',color='blue',label = f"Q={Qlst[0]} GeV")
    plt.plot(df_Q2["bT"], df_Q2["Gammatg"], marker='s',color='blue',label = f"Q={Qlst[1]} GeV")
    plt.plot(df_Q3["bT"], df_Q3["Gammatg"], marker='^',color='blue',label = f"Q={Qlst[2]} GeV")
    plt.plot(df_Q4["bT"], df_Q4["Gammatg"], marker='*',color='blue',label = f"Q={Qlst[3]} GeV")
    plt.plot(df_Q5["bT"], df_Q5["Gammatg"], marker='+',color='blue',label = f"Q={Qlst[4]} GeV")
    
    
    plt.xlabel(r"bT (GeV$^-1$)")
    plt.ylabel(r"$\tilde{\Gamma}_i$")
    plt.grid(True)
    plt.yscale("log")
    plt.legend()
    plt.show()

def Gamma_tilde_Resum_cal_plt(gammainit, Qlst, bTlst,bmax,gq,gg):
    
    Gamma_tilde = []
    for bT in bTlst:
        for Q in Qlst:
            f = Gamma_tilde_Resum_Evo(gammainit, Q, bT,bmax,gq,gg)
            Gamma_tilde.append((bT, Q, f[0],f[1]))
            
    df = pd.DataFrame(Gamma_tilde, columns=["bT", "Q", "Gammatq", "Gammatg"])
    
    df_Q1 = df[df["Q"] == Qlst[0]]
    df_Q2 = df[df["Q"] == Qlst[1]]
    df_Q3 = df[df["Q"] == Qlst[2]]
    df_Q4 = df[df["Q"] == Qlst[3]]
    df_Q5 = df[df["Q"] == Qlst[4]]
    
    plt.plot(df_Q1["bT"], df_Q1["Gammatq"], marker='o',color='red',label = f"Q={Qlst[0]} GeV")
    plt.plot(df_Q2["bT"], df_Q2["Gammatq"], marker='s',color='red',label = f"Q={Qlst[1]} GeV")
    plt.plot(df_Q3["bT"], df_Q3["Gammatq"], marker='^',color='red',label = f"Q={Qlst[2]} GeV")
    plt.plot(df_Q4["bT"], df_Q4["Gammatq"], marker='*',color='red',label = f"Q={Qlst[3]} GeV")
    plt.plot(df_Q5["bT"], df_Q5["Gammatq"], marker='+',color='red',label = f"Q={Qlst[4]} GeV")
    
    
    plt.plot(df_Q1["bT"], df_Q1["Gammatg"], marker='o',color='blue',label = f"Q={Qlst[0]} GeV")
    plt.plot(df_Q2["bT"], df_Q2["Gammatg"], marker='s',color='blue',label = f"Q={Qlst[1]} GeV")
    plt.plot(df_Q3["bT"], df_Q3["Gammatg"], marker='^',color='blue',label = f"Q={Qlst[2]} GeV")
    plt.plot(df_Q4["bT"], df_Q4["Gammatg"], marker='*',color='blue',label = f"Q={Qlst[3]} GeV")
    plt.plot(df_Q5["bT"], df_Q5["Gammatg"], marker='+',color='blue',label = f"Q={Qlst[4]} GeV")
    
    
    plt.xlabel(r"bT (GeV$^-1$)")
    plt.ylabel(r"$\tilde{\Gamma}_i$")
    plt.grid(True)
    plt.yscale("log")
    plt.legend()
    plt.show()

def EEC_cal_plt(theta_lst, Q_lst, gammainit,bmax,gq,gg,fq,fg):
    
    EEC = []
    for theta in theta_lst:
        for Q in Q_lst:
            z = dEEC(theta, Q, gammainit, bmax, gq, gg, fq, fg)
            EEC.append((theta, Q, z))

    df = pd.DataFrame(EEC, columns=["theta", "Q", "dEEC"])
    df.to_csv("dEECcal.csv", index=False)

    df = pd.read_csv("dEECcal.csv")
    
    df_Q1 = df[df["Q"] == Q_lst[0]]
    df_Q2 = df[df["Q"] == Q_lst[1]]
    df_Q3 = df[df["Q"] == Q_lst[2]]
    df_Q4 = df[df["Q"] == Q_lst[3]]
    df_Q5 = df[df["Q"] == Q_lst[4]]
    
    plt.plot(df_Q1["theta"], df_Q1["dEEC"], marker='o',label = f"Q={Q_lst[0]} GeV")
    plt.plot(df_Q2["theta"], df_Q2["dEEC"], marker='s',label = f"Q={Q_lst[1]} GeV")
    plt.plot(df_Q3["theta"], df_Q3["dEEC"], marker='^',label = f"Q={Q_lst[2]} GeV")
    plt.plot(df_Q4["theta"], df_Q4["dEEC"], marker='*',label = f"Q={Q_lst[3]} GeV")
    plt.plot(df_Q5["theta"], df_Q5["dEEC"], marker='+',label = f"Q={Q_lst[4]} GeV")
    
    plt.xlabel("|theta|")
    plt.ylabel("dEEC")
    plt.title("dEEC vs theta")
    plt.grid(True)
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    
if __name__ == '__main__':
    
    # Test of Gamma(mu)
    '''
    gammainit = np.array([0.7,0.7])
    Q1lst = np.exp(np.linspace(np.log(2),np.log(500),15))
    Gamma_cal_plt(gammainit, Q1lst)
    '''
    
    # Test of Gamma_tilde_Perturbative_Evo(mu,bT)
    '''
    gammainit = np.array([0.7,0.7])
    Qlst = np.linspace(20, 100, 5)
    bTlst = np.linspace(10. ** (-6),2,20)
    Gamma_tilde_cal_plt(gammainit,Qlst,bTlst )
    '''
    # Test of Gamma_tilde_Resum_Evo(mu,bT)
    '''
    gammainit = np.array([0.7,0.7])
    Qlst = np.linspace(20, 100, 5)
    bTlst = np.linspace(10. ** (-6),5,100)
    
    bmax = 1.5
    gq = 0.5    
    gg = 0.5
    Gamma_tilde_Resum_cal_plt(gammainit,Qlst,bTlst,bmax,gq,gg )
    '''

    gammainit=np.array([0.7,0.7])
    theta_lst = np.linspace(0, 0.4, 100)
    Q_lst = np.linspace(20, 100, 5)
    
    bmax = 1.5
    gq = 0.5
    gg = 0.5
    fq = 1
    fg = 0
    
    #dEEC(0.1, 20, gammainit, bmax, gq, gg, fq, fg)
    
    EEC_cal_plt(theta_lst, Q_lst,gammainit,bmax,gq,gg,fq,fg)
    
