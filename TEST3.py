import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import quad

# ======================================
# Load your data
# ======================================
df = pd.read_csv("Output/Gamma_thetaQ_data.csv")   # replace with your filename

# Unique Q values
Qvals = sorted(df["Q"].unique())

# ======================================
# Helper: log-log interpolation function
# ======================================
def build_log_interp(x, y):
    """
    Build log-log interpolation, but works even if zeros do not appear.
    """
    x_safe = np.array(x)
    y_safe = np.array(y)

    # Avoid any zero (your data had no zeros, but this keeps it safe)
    eps = 1e-40
    x_safe = np.maximum(x_safe, eps)
    y_safe = np.maximum(y_safe, eps)

    return interp1d(
        np.log(x_safe),
        np.log(y_safe),
        kind='linear',
        fill_value='extrapolate'
    )

# ======================================
# Integral of gamma * qT dqT
# ======================================
def compute_integrals_for_Q(dfQ):
    """
    dfQ is the slice of the dataframe for a fixed Q.
    """
    qT = dfQ["qT"].values
    gq = dfQ["gamma_q"].values
    gg = dfQ["gamma_g"].values

    # Build log–log interpolation
    fq = build_log_interp(qT, gq)
    fg = build_log_interp(qT, gg)

    # gamma(qT)
    gamma_q = lambda t: np.exp(fq(np.log(t)))
    gamma_g = lambda t: np.exp(fg(np.log(t)))

    # integrands
    integrand_q = lambda t: gamma_q(t) * t
    integrand_g = lambda t: gamma_g(t) * t

    qT_min = np.min(qT)
    qT_max = np.max(qT)

    # (1) Integral from 0 to qT_min using extrapolation
    I_q_0 = quad(integrand_q, 0, qT_min, limit=400)[0]
    I_g_0 = quad(integrand_g, 0, qT_min, limit=400)[0]

    # (2) Integral inside data range
    I_q_mid = quad(integrand_q, qT_min, qT_max, limit=2000)[0]
    I_g_mid = quad(integrand_g, qT_min, qT_max, limit=2000)[0]

    # (3) Tail integral qT_max → ∞
    # Fit exponential tail: gamma(qT) ~ a * exp(-b*qT)
    tail_range = qT[-5:]  # last few points
    gq_tail = gq[-5:]
    gg_tail = gg[-5:]

    # Fit log gamma ≈ log a – b qT
    def fit_exp_tail(qT_arr, g_arr):
        y = np.log(g_arr)
        A = np.vstack([np.ones_like(qT_arr), -qT_arr]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        log_a, b = coef
        return np.exp(log_a), b

    aq, bq = fit_exp_tail(tail_range, gq_tail)
    ag, bg = fit_exp_tail(tail_range, gg_tail)

    # analytic integral: ∫ a e^{-b t} t dt = (a/b^2) e^{-b t}(-b t - 1)
    def tail_integral(a, b, t0):
        return a * np.exp(-b*t0) * (t0/b + 1/(b*b))

    I_q_tail = tail_integral(aq, bq, qT_max)
    I_g_tail = tail_integral(ag, bg, qT_max)
    return I_q_0 + I_q_mid, I_g_0 + I_g_mid 
    #return I_q_0 + I_q_mid + I_q_tail, I_g_0 + I_g_mid + I_g_tail

# ======================================
# Compute all Q values
# ======================================
results = []

for Q in Qvals:
    dfQ = df[df["Q"] == Q]
    Iq, Ig = compute_integrals_for_Q(dfQ)
    results.append((Q, Iq, Ig))

# Turn into DataFrame
res_df = pd.DataFrame(results, columns=["Q", "Integral_gamma_q", "Integral_gamma_g"])

print(res_df)
