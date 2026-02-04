import numpy as np
from scipy.special import gammaln

def get_charlier_polynomials(x_grid, mu, K=4):
    """논문 Section 2.3: 삼항 재귀식 기반 Charlier 다항식 재현"""
    N = len(x_grid)
    C = np.zeros((N, K + 1))
    C[:, 0] = 1.0
    if K >= 1: C[:, 1] = 1.0 - (x_grid / mu)
    
    for n in range(1, K):
        # mu*C_{n+1} = (mu + n - x)*Cn - n*C_{n-1}
        C[:, n+1] = ((mu + n - x_grid) * C[:, n] - n * C[:, n-1]) / mu
        
    # Orthonormalization: psi_n = C_n / sqrt(n! / mu^n)
    psi = np.zeros_like(C)
    for n in range(K + 1):
        ln_hn = gammaln(n + 1) - n * np.log(mu)
        psi[:, n] = C[:, n] / np.sqrt(np.exp(ln_hn))
    return psi

def get_meixner_polynomials(x_grid, beta, c, K=4):
    """논문 Section 2.4: 삼항 재귀식 기반 Meixner 다항식 재현"""
    N = len(x_grid)
    M = np.zeros((N, K + 1))
    M[:, 0] = 1.0
    if K >= 1: M[:, 1] = 1.0 - (x_grid * (1.0 - c) / (beta * c))
    
    for n in range(1, K):
        # (n+1)*M_{n+1} = [(c-1)x + (n + c(n+beta))]Mn - c(n+beta-1)M_{n-1}
        term1 = ((c - 1.0) * x_grid + (n + c * (n + beta))) * M[:, n]
        term2 = c * (n + beta - 1.0) * M[:, n-1]
        M[:, n+1] = (term1 - term2) / (n + 1.0)
        
    # Orthonormalization: psi_n = M_n / sqrt(h_n)
    psi = np.zeros_like(M)
    for n in range(K + 1):
        # ln(h_n) = ln(n!) - [ln((beta)_n) + beta*ln(1-c) + n*ln(c)]
        ln_poch = gammaln(beta + n) - gammaln(beta)
        ln_hn = gammaln(n + 1) - (ln_poch + beta * np.log(1.0 - c) + n * np.log(c))
        psi[:, n] = M[:, n] / np.sqrt(np.exp(ln_hn))
    return psi