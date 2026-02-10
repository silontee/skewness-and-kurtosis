# src/baselines.py
import numpy as np
from scipy.stats import poisson, nbinom

EPS = 1e-15

def poisson_baseline(data):
    mu = float(np.mean(data))
    return mu, lambda x: poisson.pmf(x, mu)

def nb_moment_matched_params(data):
    mu = float(np.mean(data))
    var = float(np.var(data, ddof=0))
    if var <= mu + 1e-12:
        return None
    c = 1.0 - (mu / var)
    beta = (mu**2) / (var - mu)
    return beta, c

# ✅ (추가) expansions_mom.py가 import하는 함수
def nb_baseline_from_params(beta, c):
    # scipy: n=beta, p=1-c
    return lambda x: nbinom.pmf(x, beta, 1.0 - c)

# ✅ (추가) expansions_mom.py가 import하는 함수(호환용)
def nb_baseline_or_poisson(data):
    params = nb_moment_matched_params(data)
    if params is None:
        mu, wP = poisson_baseline(data)
        return ("NB(baseline MoM; Poisson approx)", {"note": "var<=mean -> Poisson approx", "mu": mu}, wP)
    beta, c = params
    wNB = nb_baseline_from_params(beta, c)
    return ("NB(baseline MoM)", {"beta": beta, "c": c}, wNB)

# ✅ main.py에서 쓰는 NB pmf 직접 생성 함수
def nb_pmf_mom_or_poisson(data, grid):
    mu = float(np.mean(data))
    params = nb_moment_matched_params(data)
    if params is None:
        p = poisson.pmf(grid, mu)
        p = p / max(p.sum(), EPS)
        return p, {"note": "var<=mean -> NB not applicable, used Poisson approx", "mu": mu}
    beta, c = params
    p = nbinom.pmf(grid, beta, 1.0 - c)
    p = p / max(p.sum(), EPS)
    return p, {"beta": beta, "c": c}

def xmax_from_data(data, q=0.999):
    data = np.asarray(data, dtype=int)
    return int(np.maximum(np.max(data), np.quantile(data, q)))
