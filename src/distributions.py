import numpy as np
from scipy.stats import poisson, nbinom

def get_poisson_baseline(data):
    """section 2.1 & 2.2"""
    mu = np.mean(data)
    return mu, lambda x: poisson.pmf(x, mu)

def get_nb_baseline(data):
    """section 2.1 & 2.2 과소분산 시 에러를 반환"""
    mu = np.mean(data)
    var = np.var(data)
    if var <= mu:
        return None, None # 과소분산은 NB 시스템 적용 불가
    
    c = 1.0 - (mu / var)
    beta = (mu**2) / (var - mu)
    # Scipy 매핑: n = beta, p = 1 - c
    return (beta, c), lambda x: nbinom.pmf(x, beta, 1.0 - c)

'''
 methond of moments 초기 세팅으로 데이터를 보고선 기초분포의 파라미터를 이미 결정해두는 관계
이렇게 해두면 theta1과 2가 0으로 세팅이 되면서 해석의 모호함이 줄어든다  왜도와 첨도만으로 모델이 좋아진 이유를
설명할수있게된다. 
''' 