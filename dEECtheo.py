from math import tau
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

# Custom handler with well-aligned "|" separators
class HandlerWithLineAndMarkerSeparators(HandlerTuple):
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        # orig_handle is a tuple of tuples: each item is (line_handle, marker_handle)
        handles = orig_handle
        n_items = len(handles)
        total_slots = 2 * n_items - 1
        slot_width = width / total_slots
        center_y = ydescent + height / 2

        artists = []

        for i in range(total_slots):
            x = xdescent + i * slot_width
            if i % 2 == 0:
                # Draw line and marker for the current item
                line_handle, marker_handle = handles[i // 2]

                # Draw line
                line = Line2D(
                    [x, x + slot_width], [center_y, center_y],
                    linestyle=line_handle.get_linestyle(),
                    color=line_handle.get_color(),
                    linewidth=line_handle.get_linewidth(),
                    transform=trans,
                )
                artists.append(line)

                # Draw marker at center of this slot
                marker = Line2D(
                    [x + slot_width / 2], [center_y],
                    marker=marker_handle.get_marker(),
                    color=marker_handle.get_color(),
                    linestyle='None',
                    markersize=marker_handle.get_markersize() or 8,
                    markerfacecolor=marker_handle.get_markerfacecolor() or marker_handle.get_color(),
                    markeredgecolor=marker_handle.get_markeredgecolor() or marker_handle.get_color(),
                    transform=trans,
                )
                artists.append(marker)
            else:
                # Separator "|"
                sep = Text(
                    x + slot_width / 2, center_y,
                    '|',
                    ha='center', va='center',
                    fontsize=fontsize + 2,
                    transform=trans
                )
                artists.append(sep)

        return artists

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

@cache
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

@cache
def adimLO(nf: int):
    
    cf = CF
    ca = CA
    
    gTqq0 = 25/6 * cf
    gTgq0 = - 7/6 * cf
    gTqg0 = - 7/15 * nf
    gTgg0 = 14/5 * ca + 2/3 * nf
    
    return np.array([[gTqq0,gTqg0],
                    [gTgq0,gTgg0]])
@cache
def adimNLO(nf: int):
    
    z2=zeta(2)
    z3=zeta(3)
    cf = CF
    ca = CA
    
    gTqq1 = (-5453/1800*cf*nf + cf**2*(-1693/48 + 24*z2 - 16*z3)
                 + ca*cf*(459/8 - 86*z2/3 + 8*z3) )
    gTgq1 = ca*cf*(-39451/5400 - 14*z2/3) + cf**2*(-2977/432 + 28*z2/3)
    gTqg1 = -833/216*cf*nf - 4/25*nf**2 + ca*nf*(619/2700 + 28*z2/15)
    gTgg1 = (12839/5400*cf*nf + ca*nf*(3803/1350 - 16*z2/3)
            + ca**2*(2158/675 + 52*z2/15 - 8*z3))
    
    return np.array([[gTqq1,gTqg1],
                    [gTgq1,gTgg1]])
@cache
def adimNNLO(nf: int):
    
    z2=zeta(2)
    z3=zeta(3)
    z4=zeta(4)
    z5=zeta(5)
    cf = CF
    ca = CA
    
    gTqq2 = ((112*z5+48*z2*z3-2083/3*z4+16153/18*z3-13105/72*z2-3049531/31104)*cf*ca**2
            +(-432*z5-208*z2*z3+8252/3*z4-19424/9*z3-16709/27*z2+20329835/15552)*cf**2*ca
            +(416*z5+224*z2*z3-6172/3*z4+10942/9*z3+11797/18*z2-17471825/15552)*cf**3
            +(146971/2700*z2-5803/45*z3+68/3*z4-25234031/1944000)*ca*cf*nf
            +(-9767/225*z2+8176/45*z3-136/3*z4-4100189/64800)*cf**2*nf
            -105799/162000*cf*nf**2)
    
    gTgq2 = ((-17093053/777600-50593/600*z2-2791/90*z3+196/3*z4)*cf*ca**2
                +(63294389/388800+123773/900*z2-3029/9*z3+511/3*z4)*cf**2*ca
                +(-647639/3888+3193/54*z2+2533/9*z3-308*z4)*cf**3
                +(-73/27*z2+182/9*z3+246767/60750)*ca*cf*nf
                +(-419593/81000+4/9*z2-28/9*z3)*cf**2*nf)
    
    gTqg2 = ((239959/13500*z2+343/45*z3-252/5*z4-1795237/1944000)*ca**2*nf
            +(34127/1350*z2+6208/75*z3-42/5*z4-3607891/38880)*ca*cf*nf
            +(-2042/225*z2-26102/225*z3+448/15*z4+9397651/97200)*cf**2*nf
            +(-554/135*z2-28/9*z3+1215691/121500)*ca*nf**2
            +(2738/675*z2-10657/4050)*cf*nf**2-172/1125*nf**3)
    
    gTgg2 = ((96*z5+64*z2*z3-2566/15*z4-23702/225*z3+66358/1125*z2-5819653/486000)*ca**3
            +(-12230737/1944000-51269/540*z2+239/9*z3+104*z4)*ca**2*nf
            +(-1700563/108000-16291/675*z2+282/5*z3)*ca*cf*nf
            +(219077/194400+2411/675*z2-28/9*z3)*cf**2*nf
            +(-18269/10125-64/9*z3+160/27*z2)*ca*nf**2+(-2611/162000-196/135*z2)*cf*nf**2)
    
    return np.array([[gTqq2,gTqg2],
                    [gTgq2,gTgg2]])

def adimsum(AS: float, nloop: int, nf: int):
    if(nloop>3):
        assert "Not implemented yet!"
    adimlst = np.array([adimLO(nf),adimNLO(nf),adimNNLO(nf)])
    aslst = np.array([AS,AS**2,AS**3])
    
    return np.einsum('n,nml->ml', aslst[:nloop], adimlst[:nloop])

def tot_xsec_sum(Q: float, nloop: int,  nf: int = 5):
    AS = AlphaS(3,5,Q)
    sigma = 4 * np.pi * AS**2/(3*Q**2)
    ASpi = AS/(2*np.pi)
    ASlst = np.array([1,ASpi,ASpi**2])
    Coef = np.array([1, -3/2*CF, 
            (3/2*CF)**2 - CF*( (123/8-11*zeta(3)) * CA - 3/8 * CF -(11/2-4*zeta(3))*nf/2)])
    return np.sum(ASlst[:nloop+1]*Coef[:nloop+1])

def betaLO(nf: int):
    ca=CA
    return 11/3*ca-2/3*nf

def betaNLO(nf: int):
    ca=CA
    cf=CF
    return 34/3*ca**2-10/3*ca*nf-2*cf*nf

def betaNNLO(nf: int):
    ca=CA
    cf=CF
    return (2857/54*ca**3-1415/54*ca**2*nf-205/18*ca*cf*nf+cf**2*nf
            +11/9*cf*nf**2+79/54*ca*nf**2)

'''
@cache
def evolop_adapt(mu: float, muinit: float, nlooplog: int, rtol=1e-10, atol=1e-12):
    
    t_vals = np.linspace(np.log(muinit), np.log(mu), 300)
    
    AS_vals = [AlphaS(3,5,np.exp(t))/(4*np.pi) for t in t_vals]
    
    AS_interp = interp1d(t_vals, AS_vals, kind='cubic', fill_value='extrapolate')

    # RHS of the evolution equation dU/dt = γ(α_s(t)) U
    def rhs(t, U_flat):
        U = U_flat.reshape((2, 2))
        ASt = AS_interp(t)
        gamma = adimsum(ASt,nlooplog,5)
        return (-2*gamma @ U).flatten()

    # Initial condition: identity matrix
    U0 = np.eye(2).flatten()

    # Integrate from t0 to t1
    sol = solve_ivp(rhs, [t_vals[0], t_vals[-1]], U0, rtol=rtol, atol=atol)

    # Final evolution operator
    U_final = sol.y[:, -1].reshape((2, 2))
    return U_final
'''
def HqHg(mu: float, nloop: int):
    cf=CF
    ca=CA
    z2=zeta(2)
    z3=zeta(3)
    z4=zeta(4)
    nf=5
    
    AS = AlphaS(3,5,mu)/(4*np.pi)
    aslst  = np.array([1,AS,AS**2])
    heelst = np.array([[1/2,0],
                       [cf * 131/16, cf * ( - 71/48 )],
                       [(( 16*z4 - 293/3*z3 - 83/2*z2 + 2386397/10368 ) * ca*cf
                        + ( -32*z4 + 254/3*z3 + 1751/72*z2 - 1105289/20736 ) * cf**2
                        + (4*z3+59/60*z2-8530817/432000) * cf*nf),
                        (( - 19/3*z3 + 47/45*z2 - 29802739/1296000 ) * ca*cf
                            + ( 31/3*z3 + 523/72*z2 - 674045/20736 ) * cf**2)]])
    
    return np.einsum('i,ij->j',aslst[:nloop],heelst[:nloop])

J = 3 - 1
NF = 5
P = 1
nloop = 3
mu0 = 2
#nlooplog = 3

def Gamma_Evo(Gamma_Init: np.array, mu: float, nlooplog: int):
    
    Evo0 = evolop(J, NF, P, mu, mu0, nloop)
    #Evo0=evolop_adapt(mu,mu0,nlooplog)
    Gamma_mu = 1 - np.einsum('i,ij->j', 1-Gamma_Init, Evo0)
    
    return Gamma_mu

@lru_cache(maxsize=None)
def _Gamma_tilde_Pert_Evo(mu: float, bT: float, nlooplog: int, Gamma_Init_tuple: tuple):
    
    Gamma_Init = np.array(Gamma_Init_tuple)
    mub = 2 * np.exp(-np.euler_gamma) / bT
    Gamma_mub = Gamma_Evo(Gamma_Init, mub, nlooplog)
    Evo = evolop(J, NF, P, mu, mub, nloop)
    #Evo = evolop_adapt(mu,mub,nlooplog)
    
    return np.einsum('i,ij->j', Gamma_mub, Evo)

def Gamma_tilde_Perturbative_Evo(Gamma_Init: np.ndarray, mu: float, bT: float, nlooplog: int):
    return _Gamma_tilde_Pert_Evo(mu, bT, nlooplog, tuple(Gamma_Init))


def Cimp(MU:float, qT: float, MU0: float, Lambda: float):

    Evo3 = evolop(J, NF, P, MU, MU0, nloop)
    Evo5 = evolop(J+2, NF, P, MU, MU0, nloop)
    
    C_evo = (np.linalg.inv(Evo3) @ Evo5) - np.eye(2)
    
    return np.eye(2) + C_evo* np.exp(-qT**2/Lambda**2)
    #return np.eye(2) + C_evo* np.exp(-qT/Lambda)

BMAX_INTEGRAL = 30

'''
def Gamma_tilde_ResumLLA(Gamma_Init: np.ndarray, Q: float, bT: float, bmax: float, gq: float, gg: float, nlooplog: int):
    bstar = bT / np.sqrt(1 + bT**2 / bmax**2)
    Gamma_Pert = Gamma_tilde_Perturbative_Evo(Gamma_Init, Q, bstar, nlooplog)
    Gamma_NonP = np.exp(-np.array([gq, gg]) * bT)
    return Gamma_Pert * Gamma_NonP

def Gamma_tilde_Resum(Gamma_Init: np.ndarray, Q: float, bmax: float, gq: float, gg: float, nlooplog: int):

    logbT = np.linspace(np.log(0.001),np.log(BMAX_INTEGRAL),100)
    bTlst = np.exp(logbT)
    muinit = 100 # Initial scale for evolution
    
    Gamma_tilde_LLA_Init = np.array([[bt,*Gamma_tilde_ResumLLA(Gamma_Init, muinit, bt, bmax, gq, gg, nlooplog)] for bt in bTlst])

    return Gamma_tilde_LLA_Init
'''

@lru_cache(maxsize=None)
def GammaRes_NLO(Q: float, bT: float, Gammaq: float, Gammag: float, bmax: float, gq: float, gg: float, nlooplog: int):
    
    Gamma_Init = np.array([Gammaq, Gammag])
    
    def gammaresLLA(bT):
        bstar = bT / np.sqrt(1 + bT**2 / bmax**2)
        Gamma_Pert = Gamma_tilde_Perturbative_Evo(Gamma_Init, Q, bstar, nlooplog)
        #Gamma_NonP = np.exp(-np.array([gq, gg]) * bT**2)
        Gamma_NonP = np.exp(-np.array([gq, gg]) * bT)
        Gamma = Gamma_Pert * Gamma_NonP
        
        return Gamma

    def xintegrand(x):
        AS = AlphaS(3,5,Q)/(2*np.pi)
        Gamma = gammaresLLA(bT)
        Gammax = gammaresLLA(bT/x)
        integrandq = (3*Gamma[0] * (1 + AS * CF * (2*np.pi**2/3-9/2)) # Integrate x^2 dx from 0 to 1= 1/3. A factor of 3 for normalization
                    +CF * AS * Gammax[0] * (2*(1+x**2)/(1-x)*np.log(x)-3/2*x+5/2)
                    +CF * AS * np.log(1-x)/(1-x) * ((1+x**2)*Gammax[0] - 2*Gamma[0])
                    -3/2 * CF * AS * 1/(1-x) * (Gammax[0] - Gamma[0]))
        
        integrandg = CF * AS * (1+(1-x)**2)/x *np.log(x**2*(1-x)) * Gammax[1]
        
        return np.array([integrandq, integrandg])* x**2/2
        
    gammaNLO, _ = quad_vec(xintegrand,0,1)
            
    return gammaNLO

def dEECNLO(theta: float, Q: float, Gamma_Init: np.array, bmax: float, gq: float, gg: float, fq: float, fg:float,nlooplog: int):
    
    fqg=np.array([fq,fg])
    Gammaq = Gamma_Init[0]
    Gammag = Gamma_Init[1]
    
    def integrand(bT):

        gammanlo = GammaRes_NLO(Q,bT, Gammaq, Gammag, bmax, gq, gg, nlooplog)
        gamma_dot = np.dot(gammanlo, fqg)
        return bT * j0(theta * bT * Q) * gamma_dot

    integral, error = quad(integrand, 0, BMAX_INTEGRAL, epsabs=1e-6, epsrel=1e-6, limit=200)
    result = integral * Q**2
    
    return result

def dEECimprov(theta: float, Q: float, Gamma_Init: np.array, bmax: float, gq: float, gg: float, fq: float, fg:float,nlooplog: int, MU0: float, Lambda: float):
    
    hqhg = HqHg(Q,nlooplog)
    fqg=np.array([fq,fg])
    fqg= fqg * hqhg
    
    cimp = Cimp(Q, theta*Q, MU0, Lambda)

    def integrand(bT):
        bstar = bT / np.sqrt(1 + bT**2 / bmax**2)
        Gamma_Pert = Gamma_tilde_Perturbative_Evo(Gamma_Init, Q, bstar, nlooplog)
        #Gamma_NonP = np.exp(-np.array([gq, gg]) * bT**2)
        Gamma_NonP = np.exp(-np.array([gq, gg]) * bT)
        Gamma = Gamma_Pert * Gamma_NonP
        gamma_dot = Gamma @ cimp @ fqg
        return bT * j0(theta * bT * Q) * gamma_dot

    integral, error = quad(integrand, 0, BMAX_INTEGRAL, epsabs=1e-6, epsrel=1e-6, limit=200)
    result = integral * Q**2
    
    return result

def dEEC(theta: float, Q: float, Gamma_Init: np.array, bmax: float, gq: float, gg: float, fq: float, fg:float,nlooplog: int):
    
    hqhg = HqHg(Q,nlooplog)
    fqg=np.array([fq,fg])
    fqg= fqg * hqhg
    
    def integrand(bT):
        bstar = bT / np.sqrt(1 + bT**2 / bmax**2)
        Gamma_Pert = Gamma_tilde_Perturbative_Evo(Gamma_Init, Q, bstar, nlooplog)
        #Gamma_NonP = np.exp(-np.array([gq, gg]) * bT**2)
        Gamma_NonP = np.exp(-np.array([gq, gg]) * bT)
        Gamma = Gamma_Pert * Gamma_NonP
        gamma_dot = np.dot(Gamma, fqg)
        return bT * j0(theta * bT * Q) * gamma_dot

    integral, error = quad(integrand, 0, BMAX_INTEGRAL, epsabs=1e-6, epsrel=1e-6, limit=200)
    result = integral * Q**2
    
    return result

def dEEC_fixed_order(z: float, mu: float, nloop_alphaS: int, nf: int, nloopEEC: int):
    AS = AlphaS(nloop_alphaS, nf, mu)/(4*np.pi)
    
    aslst = np.array([AS,AS**2,AS**3])
    
    zlst = np.array([2, 
                     -173/15* np.log(z) + 16/9*zeta(3)-424/27*zeta(2)+638941/6075,
                     20317/450 * (np.log(z))**2 + np.log(z)*(3704/81*zeta(3)-343252/1215*zeta(2)-686702711/1093500)
                     +352/27*zeta(5)+160/9*zeta(2)*zeta(3)-8930/81*zeta(4)-633376/405*zeta(3)-18994669/36450*zeta(2)+745211486777/131220000])
    
    if(nloopEEC > 3):
        assert "Not implemented!"
        
    return np.dot(aslst[:nloopEEC], zlst[:nloopEEC])/z

def dEEC_NNLL_resum(z: float, mu: float, nloop_alphaS: int, nf: int):
    
    AS = AlphaS(nloop_alphaS, nf, mu)/(4*np.pi)
    
    zdsigma_dz_ee_NNLL_SU3_5flavor_num = (2.* AS + 
        AS**2*(-11.5333333333333333333333333333*np.log(z) 
            +81.4809061031774183115800436626)      
        +AS**3*(45.1488888888888888888888888889*np.log(z)**2           
        -1037.73139012054533703493186893*np.log(z)     
            +2871.36018216039823956603951601) 
        +AS**4*(-256.025827160493827160493827160*np.log(z)**3 
            +7747.42239070965646392858858228*np.log(z)**2 
            -61715.2555242123607041254763495*np.log(z)) 
        +AS**5*(1602.60246803840877914951989026*np.log(z)**4 
            -64568.2582283506873397254346998*np.log(z)**3 
            +722515.516256038040601336185317*np.log(z)**2) 
        +AS**6*(-10534.7267712385307117817405883*np.log(z)**5 
            +536394.811477879909321558903506*np.log(z)**4 
            -8146312.10248108362757056811144*np.log(z)**3) 
        +AS**7*(71333.3362324142277317707892315*np.log(z)**6 
            -4414730.56567107479857511474780*np.log(z)**5 
            +85819711.8182380506665106637709*np.log(z)**4) 
        +AS**8*(-492759.253892475220040104796832*np.log(z)**7 
            +36027136.1681828698891852390426*np.log(z)**6
            -859877193.794471875202029814762*np.log(z)**5)
        +AS**9*(3453271.31020790020398013590861*np.log(z)**8
            -291941430.452492896551900696451*np.log(z)**7
            +8301287375.10864442506384008482*np.log(z)**6) )
    
    return zdsigma_dz_ee_NNLL_SU3_5flavor_num/z

GammaDF = pd.read_csv("ee_EEC_data/Sum_EEC_ee.txt", sep=r"\s+", header= 0, names = ['Q','f'])

def Gamma_cal_plt(gammainit, Qlst, nlooplog):

    GammaEvolst=np.array([Gamma_Evo(gammainit, Q, nlooplog)  for Q in Qlst])
    
    GammaEvolstq = GammaEvolst[:,0]
    GammaEvolstg = GammaEvolst[:,1]
    
    plt.figure(figsize=(5.5, 4)) 
    
    plt.plot(Qlst, GammaDF['f'],color='black',marker="o",linestyle="none",label = r"PYTHIA")
    plt.plot(Qlst, (1+GammaEvolstq)/2,color='magenta',linestyle='--',label = r"LO Theory")
    plt.plot(Qlst, GammaDF['f pred'],color='magenta',label = r"NLO Theory")
    

    #plt.plot(Qlst, GammaEvolstg,color='blue',label = r"$\Gamma_g$")
    #plt.plot(Qlst, 1-GammaEvolstq,linestyle='--',color='red', markerfacecolor='none',label = r"$\Gamma'_q$")
    #plt.plot(Qlst, 1-GammaEvolstg,color='blue',linestyle='--', markerfacecolor='none',label = r"$\Gamma'_g$")
    
    plt.xlim(12.5, 580) 
    plt.ylim(0.80, 1.0) 
    
    plt.title(r"Integrated EEC in $e^+e^-$ from PYTHIA and theory", fontsize = 14)
    plt.xlabel("Q (GeV)", fontsize = 13)
    plt.ylabel("$\Sigma_2$", fontsize = 13)
    plt.xscale("log")
    plt.xticks([15, 30, 50, 100, 200, 500], ["15", "30", "50", "100", "200", "500"])
    plt.grid(True)
    plt.legend(fontsize=13)
    plt.tight_layout(pad=0.1)
    plt.savefig("Output/IntegratedEEC.pdf", format="pdf") 

def Gamma_tilde_cal_plt(gammainit, Qlst, bTlst,nlooplog):
    
    Gamma_tilde = []
    for bT in bTlst:
        for Q in Qlst:
            f = Gamma_tilde_Perturbative_Evo(gammainit, Q, bT,nlooplog)
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

def EEC_Compute_Aux(args):
    theta, Q, gammainit, bmax, gq, gg, fq, fg, nlooplog = args
    z = (1 - np.cos(theta)) / 2

    f = dEEC(theta, Q, gammainit, bmax, gq, gg, fq, fg, nlooplog)
    fz = 2 * theta / np.sin(theta) * f

    fNLO = dEECNLO(theta, Q, gammainit, bmax, gq, gg, fq, fg, nlooplog)
    fzNLO = 2 * theta / np.sin(theta) * fNLO

    return (theta, Q, f, z, fz, fNLO, fzNLO)
    
def dEEC_Res_cal_plt(theta_lst, Q_lst, gammainit,bmax,gq,gg,fq,fg,nlooplog):
    
    #'''
    def parallel_EEC(theta_lst, Q_lst, gammainit, bmax, gq, gg, fq, fg, nlooplog, nproc=8):
        # Prepare all combinations of (theta, Q)
        args_list = [
            (theta, Q, gammainit, bmax, gq, gg, fq, fg, nlooplog)
            for theta in theta_lst
            for Q in Q_lst
        ]

        # Run in parallel
        with Pool() as pool:
            results = pool.map(EEC_Compute_Aux, args_list)

        return results
    
    EEC = parallel_EEC(theta_lst, Q_lst, gammainit, bmax, gq, gg, fq, fg, nlooplog)
    
    df = pd.DataFrame(EEC, columns=["theta", "Q", "dEEC","z","dEECz", "dEECNLO", "dEECzNLO"])
    df.to_csv("Output/dEECcal.csv", index=False)
    #'''
    df = pd.read_csv("Output/dEECcal.csv")
    
    df_Q1 = df[df["Q"] == Q_lst[0]]
    df_Q2 = df[df["Q"] == Q_lst[1]]
    df_Q3 = df[df["Q"] == Q_lst[2]]
    
    colors = plt.cm.tab10.colors 
    
    plt.plot(Q_lst[0]**2*df_Q1["z"], 1/Q_lst[0]**2 * df_Q1["dEEC"], color=colors[0],linestyle='-',label = f"Q={Q_lst[0]} GeV")
    plt.plot(Q_lst[1]**2*df_Q2["z"], 1/Q_lst[1]**2 * df_Q2["dEEC"], color=colors[1],linestyle='-',label = f"Q={Q_lst[1]} GeV")
    plt.plot(Q_lst[2]**2*df_Q3["z"], 1/Q_lst[2]**2 * df_Q3["dEEC"], color=colors[2],linestyle='-',label = f"Q={Q_lst[2]} GeV")

    plt.plot(Q_lst[0]**2*df_Q1["z"], 1/Q_lst[0]**2 * df_Q1["dEECNLO"], color=colors[0],linestyle='--', label = f"Q={Q_lst[0]} GeV (NLO)")
    plt.plot(Q_lst[1]**2*df_Q2["z"], 1/Q_lst[1]**2 * df_Q2["dEECNLO"], color=colors[1],linestyle='--', label = f"Q={Q_lst[1]} GeV (NLO)")
    plt.plot(Q_lst[2]**2*df_Q3["z"], 1/Q_lst[2]**2 * df_Q3["dEECNLO"], color=colors[2],linestyle='--', label = f"Q={Q_lst[2]} GeV (NLO)")
    #plt.xlim(0.002,50)
    
    plt.xlabel(r"$z\bar{E}^2$")
    plt.ylabel(r"$dEEC/(dz\bar{E}^2)$")
    plt.title(r"$dEEC/(dz\bar{E}^2)$ vs $z\bar{E}^2$")
    plt.grid(True)
    plt.legend(fontsize=12,
        loc='upper right',
        handlelength=2)
    plt.xscale("log")
    #plt.yscale("log")
    plt.savefig("Output/dEECdzQ.pdf",bbox_inches='tight')

def dEEC_cal_plt(zlst, Q_lst):
    
    EEC1 = []
    EEC2 = []
    EEC3 = []
    EEC4 = []
    for z in zlst:
        for Q in Q_lst:
            deec1 = dEEC_fixed_order(z, Q, 3,5,1)
            EEC1.append((z,Q,deec1))
            deec2 = dEEC_fixed_order(z, Q, 3,5,2)
            EEC2.append((z,Q,deec2))
            deec3 = dEEC_fixed_order(z, Q, 3,5,3)
            EEC3.append((z,Q,deec3))
            deec4 = dEEC_NNLL_resum(z, Q, 3,5)
            EEC4.append((z,Q,deec4))

    df1 = pd.DataFrame(EEC1, columns=["z", "Q", "dEEC"])
    df2 = pd.DataFrame(EEC2, columns=["z", "Q", "dEEC"])
    df3 = pd.DataFrame(EEC3, columns=["z", "Q", "dEEC"])
    df4 = pd.DataFrame(EEC4, columns=["z", "Q", "dEEC"])
    df_Resum = pd.read_csv("Output/dEECcal.csv")
    
    df1_Q1 = df1[df1["Q"] == Q_lst[0]]
    df2_Q1 = df2[df2["Q"] == Q_lst[0]]
    df3_Q1 = df3[df3["Q"] == Q_lst[0]]
    df4_Q1 = df4[df4["Q"] == Q_lst[0]]
    df_Resum_Q1 = df_Resum[df_Resum["Q"] == Q_lst[0]]
    
    df1_Q2 = df1[df1["Q"] == Q_lst[1]]
    df2_Q2 = df2[df2["Q"] == Q_lst[1]]
    df3_Q2 = df3[df3["Q"] == Q_lst[1]]
    df4_Q2 = df4[df4["Q"] == Q_lst[1]]
    df_Resum_Q2 = df_Resum[df_Resum["Q"] == Q_lst[1]]
    
    df1_Q3 = df1[df1["Q"] == Q_lst[2]]
    df2_Q3 = df2[df2["Q"] == Q_lst[2]]
    df3_Q3 = df3[df3["Q"] == Q_lst[2]]
    df4_Q3 = df4[df4["Q"] == Q_lst[2]]
    df_Resum_Q3 = df_Resum[df_Resum["Q"] == Q_lst[2]]

    fig, ax = plt.subplots()
    
    # Colors by Q value
    scales = [100., 1.0, 0.01]
    colors = ['red', 'blue', 'green']
    Q_labels = [f"Q={Q} GeV" for Q in Q_lst]

    # --- Plot actual curves ---
    # LO
    ax.plot(df1_Q1["z"],scales[0]* df1_Q1["dEEC"], linestyle=':', color=colors[0])
    ax.plot(df1_Q2["z"],scales[1]* df1_Q2["dEEC"], linestyle=':', color=colors[1])
    ax.plot(df1_Q3["z"],scales[2]* df1_Q3["dEEC"], linestyle=':', color=colors[2])

    # NLO
    ax.plot(df2_Q1["z"],scales[0]* df2_Q1["dEEC"], linestyle='-.', color=colors[0])
    ax.plot(df2_Q2["z"],scales[1]* df2_Q2["dEEC"], linestyle='-.', color=colors[1])
    ax.plot(df2_Q3["z"],scales[2]* df2_Q3["dEEC"], linestyle='-.', color=colors[2])

    # NNLO
    ax.plot(df3_Q1["z"],scales[0]* df3_Q1["dEEC"], linestyle='--', color=colors[0])
    ax.plot(df3_Q2["z"],scales[1]* df3_Q2["dEEC"], linestyle='--', color=colors[1])
    ax.plot(df3_Q3["z"],scales[2]* df3_Q3["dEEC"], linestyle='--', color=colors[2])

    # NNLL
    ax.plot(df4_Q1["z"],scales[0]* df4_Q1["dEEC"], linestyle='-', color=colors[0])
    ax.plot(df4_Q2["z"],scales[1]* df4_Q2["dEEC"], linestyle='-', color=colors[1])
    ax.plot(df4_Q3["z"],scales[2]* df4_Q3["dEEC"], linestyle='-', color=colors[2])

    # TMD
    ax.plot(df_Resum_Q1["z"],scales[0]* df_Resum_Q1["dEECz"], marker='o', linestyle='', color=colors[0], markersize = 7, markerfacecolor='none')
    ax.plot(df_Resum_Q2["z"],scales[1]* df_Resum_Q2["dEECz"], marker='o', linestyle='', color=colors[1], markersize = 7, markerfacecolor='none')
    ax.plot(df_Resum_Q3["z"],scales[2]* df_Resum_Q3["dEECz"], marker='o', linestyle='', color=colors[2], markersize = 7, markerfacecolor='none')


    def grouplegend(colors, linesty, markersty):
        
        group = (
        (Line2D([0], [0], color=colors[0], linestyle=linesty, linewidth=1), Line2D([0], [0], color=colors[0], marker=markersty, linestyle='None', markersize = 6, markerfacecolor='none')),
        (Line2D([0], [0], color=colors[1], linestyle=linesty, linewidth=1), Line2D([0], [0], color=colors[1], marker=markersty, linestyle='None', markersize = 6, markerfacecolor='none')),
        (Line2D([0], [0], color=colors[2], linestyle=linesty, linewidth=1), Line2D([0], [0], color=colors[2], marker=markersty, linestyle='None', markersize = 6, markerfacecolor='none'))
        )
        return group
    
    Linestylst=[":","-.","--","-",""]
    Makerstylst=["","","","","o"]
    grouplegendlst = [grouplegend(colors,x,y) for x, y in zip(Linestylst, Makerstylst)]
    legendlst=["LO","NLO","NNLO","NNLL resum","TMD resum"]
    
    class SmallSquare(Rectangle):
        def __init__(self, facecolor, edgecolor, label, size=10):
            # size in points, converted to display units inside legend
            super().__init__((0,0), width=size, height=size,
                            facecolor=facecolor, edgecolor=edgecolor, label=label)
        
    square_handles  = [
    SmallSquare(facecolor=colors[0], edgecolor='white', label=fr'$\bar E$ = {Q_lst[0]} GeV'+rf'$(\times {scales[0]})$',size=4),
    SmallSquare(facecolor=colors[1], edgecolor='white', label=fr'$\bar E$ = {Q_lst[1]} GeV',size=4),
    SmallSquare(facecolor=colors[2], edgecolor='white', label=fr'$\bar E$ = {Q_lst[2]} GeV'+rf'$(\times {scales[2]})$',size=4),
    ]
    
    leg1 = ax.legend(
        handles=square_handles,
        fontsize=11,
        loc='lower left',   
        frameon=True,
        handlelength=1.5
    )
    
    ax.legend(
        grouplegendlst,
        legendlst,
        handler_map={tuple: HandlerWithLineAndMarkerSeparators()},
        fontsize=11,
        loc='upper right',
        handlelength=4
        )
    
    ax.add_artist(leg1)
    
    ax.set_xlabel("z",fontsize=13)
    ax.set_ylabel(r"$dEEC/dz$",fontsize=13)
    ax.set_title(r"$dEEC/dz$ vs z compared with the collinear ones",fontsize=14)
    ax.grid(True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    plt.savefig("Output/dEECcompare35.pdf",bbox_inches='tight')

if __name__ == '__main__':
    
    # Test of Gamma(mu)
    #'''
    Q1lst = np.array(GammaDF['Q'])
    
    def gammafit(gammaq,gammag):
        gint=np.array([gammaq,gammag])
        GammaEvolst=np.array([Gamma_Evo(gint, Q, 1)  for Q in Q1lst])
        CF = 4/3
        GammaEvolstq = GammaEvolst[:,0]
        GammaEvolstg = GammaEvolst[:,1]

        Aslst = np.array([AlphaS(3,5,Q) for Q in Q1lst])
        
        GammaDF['f pred'] = (1-Aslst/(2*np.pi)*3/2*CF)*(1/2 + Aslst/(4*np.pi)*CF*(-89/24)
                +1/2 * GammaEvolstq + Aslst/(4*np.pi)* (131/12*GammaEvolstq-71/36*GammaEvolstg))
        
        return np.sum((np.array(GammaDF['f'])- GammaDF['f pred']) **2)
    '''
    m = Minuit(gammafit, gammaq=0.5, gammag=0.5)  # initial guesses
    m.errordef = Minuit.LEAST_SQUARES  # 1 for least-squares cost

    # Run the minimization

    m.migrad()
    m.limits['gammaq']=(0.01,0.99)
    m.limits['gammag']=(0.01,0.99)
    # Print fit results
    print(m.values)   # best-fit parameters
    print(m.errors)  
    '''
    gammainit = np.array([0.754,0.824])
    gammafit(0.754,0.824)
    Gamma_cal_plt(gammainit, Q1lst,1)
    #'''
    
    # Test of Gamma_tilde_Perturbative_Evo(mu,bT)
    '''
    gammainit = np.array([0.7,0.7])
    Qlst = np.linspace(20, 100, 5)
    bTlst = np.linspace(10. ** (-6),2,20)
    Gamma_tilde_cal_plt(gammainit,Qlst,bTlst )
    '''
    
    '''
    gammainit=np.array([0.754,0.824])
    theta_lst = 2*np.exp(np.linspace(np.log(10**(-4)), np.log(0.7), 30))
    Qlst = np.array([50.,100.,200.])

    #c = 0.346*0.5
    
    bmax = 1.5
    gq = 1.1842
    gg = 1.0

    fq = 1.0
    fg = 0.0
    nlooplog=1
    dEEC_Res_cal_plt(theta_lst, Qlst,gammainit,bmax,gq,gg,fq,fg,1)
    '''
    
    
    '''
    zlst = np.exp(np.linspace(np.log(10**(-8)), np.log(0.5), 40))
    Qlst = np.array([50.,100.,200.])
    dEEC_cal_plt(zlst,Qlst)
    '''
 
