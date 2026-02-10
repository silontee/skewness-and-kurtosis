# src/expansions_mom.py
import numpy as np
from scipy.special import gammaln
from src.moments import factorial_moments, central_moments
from src.baselines import poisson_baseline, nb_moment_matched_params, nb_baseline_from_params
from src.orthopoly import get_charlier_psi, get_meixner_psi

EPS = 1e-15

def normalize_pmf(p):
    p = np.asarray(p, dtype=float)
    p[p < 0] = 0.0
    s = p.sum()
    return p / s if s > 0 else np.ones_like(p) / len(p)

# ---- PC theta (mean-matching => theta1=0)
def pc_theta_mom(data, K=4):
    mu, _ = poisson_baseline(data)
    # 논문의 정의: theta_n = E[psi_n(X)] (Eq 22)
    # 데이터를 직접 다항식에 넣어서 평균을 구하는 것이 가장 정확합니다.
    psi_at_data = get_charlier_psi(data, mu, K=K)
    theta = np.mean(psi_at_data, axis=0)
    theta[1] = 0.0 # Mean-matching 제약
    return mu, theta

# ---- Meixner theta (mean&var matching => theta1=theta2=0)
def meixner_theta_mom(data, K=4):
    params = nb_moment_matched_params(data)
    if params is None: return None, None
    beta, c = params
    
    psi_at_data = get_meixner_psi(data, beta, c, K=K)
    theta = np.mean(psi_at_data, axis=0)
    theta[1] = 0.0 # Mean-matching
    theta[2] = 0.0 # Var-matching
    return (beta, c), theta

def build_tilt_pmf(grid, w_vals, psi, theta):
    # p(x) = w(x) * (1 + sum_{n>=1} theta_n * psi_n(x))
    z = 1.0 + psi[:, 1:] @ theta[1:]
    raw = w_vals * z
    return normalize_pmf(raw)

def fit_pc_pmf(data, grid, K=4):
    mu, w = poisson_baseline(data)
    w_vals = w(grid)
    psi = get_charlier_psi(grid, mu, K=K)
    _, theta = pc_theta_mom(data, K=K)  # theta1=0
    pmf = build_tilt_pmf(grid, w_vals, psi, theta)
    return pmf, {"mu": mu, "theta": theta.tolist()}

def fit_meixner_pmf(data, grid, K=4):
    params, theta = meixner_theta_mom(data, K=K)
    if params is None:
        return None, {"note": "underdispersion -> NBM not applicable"}
    beta, c = params
    w = nb_baseline_from_params(beta, c)
    w_vals = w(grid)
    psi = get_meixner_psi(grid, beta, c, K=K)
    pmf = build_tilt_pmf(grid, w_vals, psi, theta)
    return pmf, {"beta": beta, "c": c, "theta": theta.tolist()}
