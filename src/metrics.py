# src/metrics.py
import numpy as np
import math

EPS = 1e-15

def empirical_pmf(data, xmax):
    x = np.asarray(data, dtype=int)
    freq = np.bincount(x, minlength=xmax + 1).astype(float)
    return freq / freq.sum()

def l1_sum_abs(emp_p, model_p):
    m = min(len(emp_p), len(model_p))
    return float(np.sum(np.abs(emp_p[:m] - model_p[:m])))

def loglik_from_sample(data, model_pmf):
    x = np.asarray(data, dtype=int)
    x = np.clip(x, 0, len(model_pmf) - 1)
    return float(np.sum(np.log(np.maximum(model_pmf[x], EPS))))

def aic_bic(ll, k, n):
    aic = 2*k - 2*ll
    bic = math.log(max(n, 1))*k - 2*ll
    return aic, bic
