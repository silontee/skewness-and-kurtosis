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

# -----------------------
# Charlier polynomials
# C_n(x; mu) with orthonormalization:
# h_n = n! / mu^n,  psi_n = C_n / sqrt(h_n) = C_n * sqrt(mu^n / n!)
# 재귀식 계산과 직교 정규화해주기
# -----------------------

def get_charlier_psi(grid, mu, K=4):
    """논문 Eq (23) Charlier 다항식 + p.7 직교관계 h_n=n!/μ^n 기반 정규직교화"""
    mu = float(mu)
    x = np.asarray(grid)
    C = np.zeros((len(x), K + 1))
    C[:, 0] = 1.0
    if K >= 1: C[:, 1] = 1.0 - (x / mu)

    for n in range(1, K):
        # Charlier 3항 재귀식 (p.7 생성함수에서 유도): μC_{n+1} = (μ+n-x)C_n - nC_{n-1}
        C[:, n+1] = ((mu + n - x) * C[:, n] - n * C[:, n-1]) / mu

    psi = np.zeros_like(C)
    for n in range(K + 1):
        # p.7 직교관계: h_n = n!/μ^n, φ_n = √(μ^n/n!) C_n
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
    """논문 Eq (28) Meixner 다항식 + p.8 직교관계 h_n=n!c^{-n}/(β)_n 기반 정규직교화"""
    x = np.asarray(grid)
    M = np.zeros((len(x), K + 1))
    M[:, 0] = 1.0
    if K >= 1: M[:, 1] = 1.0 - (x * (1 - c) / (beta * c))

    for n in range(1, K):
        # Meixner 3항 재귀식 (DLMF 표준): c(β+n)M_{n+1} = [(1+c)n+cβ-(1-c)x]M_n - nM_{n-1}
        term1 = (n + (n + beta) * c - x * (1 - c)) * M[:, n]
        term2 = n * M[:, n-1]
        M[:, n+1] = (term1 - term2) / ((beta + n) * c)

    psi = np.zeros_like(M)
    for n in range(K + 1):
        # p.8 직교관계: h_n = n! c^{-n} / (β)_n, φ_n = M_n/√h_n
        h_n = (math.factorial(n) * (c**-n)) / _rising(beta, n)
        psi[:, n] = M[:, n] / np.sqrt(h_n)
    return psi
