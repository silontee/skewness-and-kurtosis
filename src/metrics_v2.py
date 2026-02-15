# src/metrics.py
import numpy as np
import math
from scipy.stats import kurtosis

EPS = 1e-15
# 원본데이터 확률분포화
def empirical_pmf(data, xmax):
    x = np.asarray(data, dtype=int)
    freq = np.bincount(x, minlength=xmax + 1).astype(float)
    return freq / freq.sum()

# L1오차 계산
def l1_sum_abs(emp_p, model_p):
    m = min(len(emp_p), len(model_p))
    return float(np.sum(np.abs(emp_p[:m] - model_p[:m])))

# 기초 통계량 및 Heavy Tail 여부 계산
def calculate_data_stats(data):
    mu = np.mean(data)
    var = np.var(data)
    # fisher=True: 정규분포의 첨도를 0으로 설정 (Positive = Heavy Tail)
    kurt = kurtosis(data, fisher=True)     
    return {
        "n": len(data),
        "mu": round(mu, 4),
        "var": round(var, 4),
        "vm": round(var / mu, 4) if mu > 0 else 0,
        "kurtosis": round(kurt, 4),
        "heavy_tail": "Yes" if kurt > 0 else "No"
    }