# 직교다항식 생성
import numpy as np
import math

#---------------------
#수학적 보조 도구
#---------------------

EPS = 1e-15
#상승계승
def _rising(a, k):
    """(a)_k rising factorial"""
    out = 1.0
    for i in range(k):
        out *= (a + i)
    return out
#하강계승
def _falling(a, k):
    """(a)_k falling factorial for integer-like a (used with -x)"""
    out = 1.0
    for i in range(k):
        out *= (a - i)
    return out

# -----------------------
# Charlier polynomials
# C_n(x; mu) with orthonormalization:
# psi_n = C_n / sqrt(n! * mu^n)
# 재귀식 계산과 직교 정규화해주기
# -----------------------

def get_charlier_psi(grid, mu, K=4):
    """논문 Eq (11), (12) 재현"""
    mu = float(mu)
    x = np.asarray(grid)
    C = np.zeros((len(x), K + 1))
    C[:, 0] = 1.0
    if K >= 1: C[:, 1] = 1.0 - (x / mu)  # 논문과 동일하게 Scaled
    
    for n in range(1, K):
        # 논문 Eq (11) 재귀식: C_{n+1} = (mu + n - x)Cn - n*Cn-1 / mu
        C[:, n+1] = ((mu + n - x) * C[:, n] - n * C[:, n-1]) / mu
        
    psi = np.zeros_like(C)
    for n in range(K + 1):
        # 논문 Eq (12): h_n = n! / mu^n
        h_n = math.factorial(n) / (mu**n)
        psi[:, n] = C[:, n] / np.sqrt(h_n)
    return psi

# -----------------------
# Meixner polynomials
# M_n(x; beta, c) = 2F1(-n, -x; beta; 1 - 1/c)
# For integer n (<=4) => finite sum k=0..n:
# M_n(x) = Σ_{k=0..n} [ (-n)_k (-x)_k / ((beta)_k k!) ] * z^k
# where z = 1 - 1/c
#
# Orthonormalization from paper:
# h_n = n! * c^{-n} / (beta)_n
# psi_n = M_n / sqrt(h_n)
# -----------------------

def get_meixner_psi(grid, beta, c, K=4):
    """논문 Eq (14), (15) 재현"""
    x = np.asarray(grid)
    M = np.zeros((len(x), K + 1))
    M[:, 0] = 1.0
    if K >= 1: M[:, 1] = 1.0 - (x * (1 - c) / (beta * c))
    
    for n in range(1, K):
        # 논문 Eq (14) 재귀식 사용 (Summation보다 훨씬 안정적)
        term1 = (n + (n + beta) * c - x * (1 - c)) * M[:, n]
        term2 = n * M[:, n-1]
        M[:, n+1] = (term1 - term2) / (beta + n)
        
    psi = np.zeros_like(M)
    for n in range(K + 1):
        # 논문 Eq (15) 하단 h_n 정의
        # h_n = n! * c^-n / (beta)_n
        h_n = (math.factorial(n) * (c**-n)) / _rising(beta, n)
        psi[:, n] = M[:, n] / np.sqrt(h_n)
    return psi
