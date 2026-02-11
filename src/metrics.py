# src/metrics.py
import numpy as np
import math

EPS = 1e-15
# 원본데이터 확률분포화
def empirical_pmf(data, xmax):
    x = np.asarray(data, dtype=int)
    freq = np.bincount(x, minlength=xmax + 1).astype(float)
    return freq / freq.sum()

#L1오차 계산
def l1_sum_abs(emp_p, model_p):
    m = min(len(emp_p), len(model_p))
    return float(np.sum(np.abs(emp_p[:m] - model_p[:m])))


