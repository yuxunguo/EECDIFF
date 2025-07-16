EEC Resummed Calculation with di-hadron fragmentation functions

This Python module provides a complete pipeline for computing **resummed Energy-Energy Correlators (EECs)** using di-hadron fragmentation functions

---

## 🔧 Main Functions Overview

This code follows a layered approach to QCD evolution, starting from the one-loop evolution kernel to fully resummed observables in angle space.

### ✅ `evolop(j, nf, p, mu, mu_init, nloop)`
Implements the **standard one-loop evolution operator** in Mellin space.  
- This function is **universal** and **tested**.
- Depends on `singlet_LO()`, `lambdaf()`, and `projectors()`.

---

### 🔁 `Gamma_Evo(Gamma_Init, mu)`
Evolves the quark/gluon di-hadron FFs moments `Gamma` from an initial scale `mu0 = 2 GeV` to a new scale `mu`.

Uses the identity:
\[
\Gamma(\mu) = 1 - \left[1 - \Gamma(\mu_0)\right] \cdot E(\mu, \mu_0)
\]
This reformulation simplifies evolution by working with $\Gamma' = 1 - \Gamma$.

---

### 📈 `Gamma_tilde_Perturbative_Evo(Gamma_Init, mu, bT)`
Computes the **perturbative part** of the resummed di-hadron FFs moments $\tilde{\Gamma}(Q, b_T)$ using:

1. Evolution of `Gamma` from $\mu_0 \rightarrow \mu_b = \frac{2 e^{-\gamma_E}}{b_T}$
2. Resummation of large logs between $\mu_b \rightarrow \mu$

---

### 📉 `Gamma_tilde_Resum_Evo(Gamma_Init, mu, bT, bmax, gq, gg)`
Applies **non-perturbative resummation** via the $b_*$ prescription:

- Regulates $b_T$ using:
\[
b_* = \frac{b_T}{\sqrt{1 + \frac{b_T^2}{b_{\text{max}}^2}}}
\]
- Adds Gaussian suppression in $b_T$ space:
\[
\tilde{\Gamma}^{\text{resum}} = \tilde{\Gamma}^{\text{pert}}(b_*) \cdot \exp(-g_i b_T^2)
\]

---

### 🔄 `dEEC(theta, Q, ...)`
Computes the **2D Fourier-Bessel transform** of $\tilde{\Gamma}^{\text{resum}}$:

\[
\text{dEEC}(\theta, Q) = \int db_T \, b_T \, J_0(\theta Q b_T) \, \tilde{\Gamma}(Q, b_T)
\]

This corresponds to the angular **integrated** EEC observable.

---

## 📊 Visualization Tools

- `Gamma_cal_plt`: Plots $\Gamma_i(Q)$ and $1 - \Gamma_i(Q)$ as functions of $Q$
- `Gamma_tilde_cal_plt`: Plots $\tilde{\Gamma}_i(Q, b_T)$ (perturbative only)
- `Gamma_tilde_Resum_cal_plt`: Plots fully resummed $\tilde{\Gamma}_i(Q, b_T)$
- `EEC_cal_plt`: Computes and plots $d\text{EEC}(\theta, Q)$ vs $\theta$

All plots support legends, log scales, and multiple $Q$ values.

---

## 🧪 Example (in `__main__`)  

You can activate and run the examples by uncommenting the relevant blocks in the `__main__` section. For example:

```python
gammainit = np.array([0.7, 0.7])
theta_lst = np.exp(np.linspace(np.log(1e-4), np.log(0.1), 100))
Q_lst = np.linspace(20, 100, 5)

bmax = 1.5
gq = 0.5
gg = 0.5
fq = 1
fg = 0

EEC_cal_plt(theta_lst, Q_lst, gammainit, bmax, gq, gg, fq, fg)
