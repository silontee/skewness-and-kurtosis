# src/orthopoly.py
import numpy as np
import math

EPS = 1e-15

def _rising(a, k):
    """(a)_k rising factorial"""
    out = 1.0
    for i in range(k):
        out *= (a + i)
    return out

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
# -----------------------
def charlier_Cn(grid, mu, n):
    """
    Compute Charlier polynomial C_n(x; mu) for all x in grid.
    Use stable recurrence:
      C_0 = 1
      C_1 = x - mu
      C_{n+1} = (x - n - mu) C_n - mu * n * C_{n-1}
    """
    x = np.asarray(grid, dtype=float)
    mu = float(mu)

    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x - mu

    Cnm1 = np.ones_like(x)      # C_0
    Cn = x - mu                 # C_1
    for k in range(1, n):
        Cnp1 = (x - k - mu) * Cn - (mu * k) * Cnm1
        Cnm1, Cn = Cn, Cnp1
    return Cn

def get_charlier_psi(grid, mu, K=4):
    """논문 Eq (11), (12) 완벽 재현"""
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
def meixner_Mn(grid, beta, c, n):
    x = np.asarray(grid, dtype=float)
    beta = float(beta)
    c = float(c)

    z = 1.0 - 1.0 / max(c, EPS)

    # finite sum
    out = np.zeros_like(x, dtype=float)
    for k in range(0, n + 1):
        # (-n)_k = (-1)^k * n!/(n-k)!  (falling factorial n*(n-1)*... )
        negn_poch = ((-1.0) ** k) * (math.factorial(n) / math.factorial(n - k))
        # (-x)_k = (-1)^k * (x)_k_falling  but x is vector (not integer always),
        # for our case x are integers grid, so we compute product (x)(x-1)...(x-k+1)
        negx_poch = ((-1.0) ** k) * _falling(x, k)

        beta_poch = _rising(beta, k)
        term = (negn_poch * negx_poch) / (max(beta_poch, EPS) * math.factorial(k))
        out += term * (z ** k)
    return out

def get_meixner_psi(grid, beta, c, K=4):
    """논문 Eq (14), (15) 완벽 재현"""
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
