import numpy as np
from scipy.special import gammaln
from src.baselines import poisson_baseline, nb_moment_matched_params, nb_baseline_from_params
from src.orthopoly import get_charlier_psi, get_meixner_psi

EPS = 1e-15
# 음수 확률값을 0으로 강제 고정
def normalize_pmf(p):
    p = np.asarray(p, dtype=float)
    p[p < 0] = 0.0
    s = p.sum()
    return p / s if s > 0 else np.ones_like(p) / len(p)

# ---- PC theta  적률법에 맞게 생성 theta1 = 0
def pc_theta_mom(data, K=4):
    mu, _ = poisson_baseline(data)
    # 논문의 정의: theta_n = E[psi_n(X)] (Eq 22)
    # 데이터를 직접 다항식에 넣어서 평균을 구하는 것이 가장 정확하다.
    psi_at_data = get_charlier_psi(data, mu, K=K)
    theta = np.mean(psi_at_data, axis=0)
    theta[1] = 0.0 
    return mu, theta

# ---- Meixner theta theta1,2 = 0
def meixner_theta_mom(data, K=4):
    params = nb_moment_matched_params(data)
    if params is None: return None, None
    beta, c = params
    
    psi_at_data = get_meixner_psi(data, beta, c, K=K)
    theta = np.mean(psi_at_data, axis=0)
    theta[1] = 0.0 
    theta[2] = 0.0 
    return (beta, c), theta

#--tilting 최종 공식
def build_tilt_pmf(grid, w_vals, psi, theta):
    z = 1.0 + psi[:, 1:] @ theta[1:]
    raw = w_vals * z
    return normalize_pmf(raw)


# Charlier,Maxiner tilt 최적화
# 소분산 방지와 파라미터 딕셔너리화
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
