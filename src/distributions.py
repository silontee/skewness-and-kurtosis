import numpy as np
from scipy.stats import poisson, nbinom

def get_poisson_baseline(data):
    """평균 매칭 (Eq 21)"""
    mu = np.mean(data)
    return mu, lambda x: poisson.pmf(x, mu)

def get_nb_baseline(data):
    """Eq (29) 기반 파라미터 매칭. 과소분산 시 에러를 반환하여 엄밀성 유지."""
    mu = np.mean(data)
    var = np.var(data)
    if var <= mu:
        return None, None # 과소분산은 NB 시스템 적용 불가 (비판적 처리)
    
    c = 1.0 - (mu / var)
    beta = (mu**2) / (var - mu)
    # Scipy 매핑: n = beta, p = 1 - c
    return (beta, c), lambda x: nbinom.pmf(x, beta, 1.0 - c)