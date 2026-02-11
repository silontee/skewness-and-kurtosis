# src/analysis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from src.orthopoly import get_charlier_psi
from src.expansions_mom import normalize_pmf
from src.metrics import l1_sum_abs

def run_convergence_study(data, grid, emp, mu, orders=[0, 4, 6, 8]):
    """차수별 수렴도를 분석하여 데이터프레임과 PMF들을 반환"""
    base_pois = stats.poisson.pmf(grid, mu)
    psi_grid = get_charlier_psi(grid, mu, K=max(orders))
    
    # 데이터 전체 범위에서 계수 추출 (Eq 17: θ*_n = E_p[ψ_n(X)])
    psi_at_data = get_charlier_psi(np.arange(np.max(data) + 1), mu, K=max(orders))
    theta_all = np.mean(psi_at_data[data], axis=0)
    theta_all[1] = 0.0  # Proposition 2: 평균 매칭(μ=E[X]) 시 θ₁=0
    
    order_rows, order_pmfs = [], {}
    for K in orders:
        tilt = 1.0 + (psi_grid[:, 1:K+1] @ theta_all[1:K+1])
        p_k = normalize_pmf(base_pois * tilt)
        order_rows.append({
            "model": f"PC-Order {K}", 
            "L1_diff": l1_sum_abs(emp, p_k), 
            "theta": theta_all[:K+1]
        })
        order_pmfs[K] = p_k
    return pd.DataFrame(order_rows), order_pmfs

def plot_comparison(name, grid, emp, model_dict, save_path):
    """표준 4대 모델 비교 그래프 (FIFA, Insurance, Simul_Heavy 용)"""
    plt.figure(figsize=(11, 7))
    plt.bar(grid, emp, width=1.0, alpha=0.3, label="Empirical", color="#2c3e50", edgecolor="black")
    
    styles = {
        "Poisson": {"color": "#95a5a6", "ls": "--", "lw": 1.5, "offset": -0.2},
        "NB":      {"color": "#e67e22", "ls": "--", "lw": 1.5, "offset": -0.07},
        "PC":      {"color": "#2980b9", "ls": "-",  "lw": 2.2, "offset": 0.07},
        "NBM":     {"color": "#c0392b", "ls": "-",  "lw": 2.2, "offset": 0.2}
    }
    for label, st in styles.items():
        if label in model_dict:
            plt.plot(grid + st["offset"], model_dict[label], label=label, 
                     color=st["color"], ls=st["ls"], lw=st["lw"], marker='o', ms=4)
            
    plt.title(f"Standard Model Comparison: {name}", fontsize=15, fontweight='bold')
    plt.legend(); plt.grid(axis='y', alpha=0.3, linestyle=':'); plt.tight_layout()
    plt.savefig(save_path, dpi=300); plt.close()

def plot_pc_convergence(name, grid, emp, order_pmfs, l1_results, save_path):
    """PC 차수별 수렴 그래프 (Insurance, Simul_Success 용)"""
    plt.figure(figsize=(11, 7))
    plt.bar(grid, emp, width=1.0, alpha=0.15, color='gray', edgecolor="#333333", label='Empirical Data')
    
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(order_pmfs)))
    for i, (K, pmf) in enumerate(order_pmfs.items()):
        l1_val = l1_results[l1_results['model'] == f"PC-Order {K}"]['L1_diff'].values[0]
        label = "Poisson (Base)" if K == 0 else f"PC Order {K}"
        plt.plot(grid, pmf, label=f"{label} (L1: {l1_val:.4f})", 
                 color=colors[i], ls='-' if K == 8 else '--', lw=3 if K == 8 else 1.8)
        
    plt.title(f"PC Expansion Convergence: {name}", fontsize=15, fontweight='bold')
    plt.legend(); plt.grid(axis='y', alpha=0.2); plt.tight_layout()
    plt.savefig(save_path, dpi=300); plt.close()