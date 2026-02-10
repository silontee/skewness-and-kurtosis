# src/moments.py
import numpy as np

def falling_factorial(x, k):
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    for j in range(k):
        out *= (x - j)
    return out

def factorial_moments(sample, max_k=4):
    x = np.asarray(sample, dtype=float)
    return {k: float(np.mean(falling_factorial(x, k))) for k in range(1, max_k + 1)}

def central_moments(sample):
    x = np.asarray(sample, dtype=float)
    mu = float(np.mean(x))
    c = x - mu
    m2 = float(np.mean(c**2))
    m3 = float(np.mean(c**3))
    m4 = float(np.mean(c**4))
    return mu, m2, m3, m4
