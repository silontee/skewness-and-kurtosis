import numpy as np
from scipy.stats import poisson, nbinom

EPS = 1e-15
#포아송 데이터 평균 추출
def poisson_baseline(data):
    mu = float(np.mean(data))
    return mu, lambda x: poisson.pmf(x, mu)

#음이항 분포 파라미터 설정
def nb_moment_matched_params(data):
    mu = float(np.mean(data))
    var = float(np.var(data, ddof=0))
    if var <= mu + 1e-12:
        return None
    c = 1.0 - (mu / var)
    beta = (mu**2) / (var - mu)
    return beta, c

# scipy함수가 논문과 파라미터가 맞지 않아서 사용
def nb_baseline_from_params(beta, c):
    # scipy: n=beta, p=1-c
    return lambda x: nbinom.pmf(x, beta, 1.0 - c)

# pmf 생성 
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


# 데이터 0.999까지 반영
def xmax_from_data(data, q=0.999):
    data = np.asarray(data, dtype=int)
    return int(np.maximum(np.max(data), np.quantile(data, q)))
