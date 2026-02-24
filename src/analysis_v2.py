# src/analysis.py
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import scipy.stats as stats
from src.orthopoly import get_meixner_psi
from src.expansions_mom import normalize_pmf
from src.metrics import l1_sum_abs
from src.baselines import nb_moment_matched_params, nb_baseline_from_params

def run_nbm_convergence_study(data, grid, emp, orders=list(range(15))):
    """
    NBM 차수별 수렴도를 분석하여 데이터프레임만 반환
    """
    params = nb_moment_matched_params(data)
    if params is None:
        return None
    beta, c = params
    
    # 기저 분포(K=0) 생성 및 정규화
    base_nb = nb_baseline_from_params(beta, c)(grid)
    base_nb /= (base_nb.sum() + 1e-15)

    max_k = max(orders)
    psi_grid = get_meixner_psi(grid, beta, c, K=max_k)
    psi_at_data = get_meixner_psi(data, beta, c, K=max_k)
    
    # 전역 계수 추출 (θ₁=0, θ₂=0 고정)
    theta_all = np.mean(psi_at_data, axis=0)
    if len(theta_all) > 1: theta_all[1] = 0.0
    if len(theta_all) > 2: theta_all[2] = 0.0
    
    order_rows, order_pmfs = [], {}
    for K in orders:
        # K차까지의 틸팅 적용
        tilt = 1.0 + (psi_grid[:, 1:K+1] @ theta_all[1:K+1])
        p_k = normalize_pmf(base_nb * tilt)
        
        err = l1_sum_abs(emp, p_k)
        order_rows.append({
            "model": f"NBM-Order {K}",
            "L1_diff": round(err, 6), 
            "beta": round(beta, 3),
            "c": round(c, 3), 
            "theta": list(np.round(theta_all[:K+1], 4))
        })
        order_pmfs[K] = p_k
        
    return pl.DataFrame(order_rows), order_pmfs

def plot_nbm_l1_convergence(name, df_conv, save_path):
    """
    X축: 차수(K), Y축: L1 Discrepancy를 시각화하는 함수
    """
    plt.figure(figsize=(8, 5))
    
    k_vals = [int(m.split()[-1]) for m in df_conv.get_column("model")]
    l1_vals = df_conv.get_column("L1_diff").to_numpy()
    
    plt.plot(k_vals, l1_vals, marker='o', linestyle='-', 
             color='#d35400', lw=2, ms=7, label='NBM L1 Error')
    
    # 차수별 수치 표시
    for row in df_conv.iter_rows(named=True):
        k_val = int(row['model'].split()[-1])
        plt.annotate(f"{row['L1_diff']:.4f}", (k_val, row['L1_diff']), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    plt.title(f"NBM Convergence L1 Error Trend: {name}", fontsize=14, fontweight='bold')
    plt.xlabel("Order of Expansion (K)", fontsize=11)
    plt.ylabel("L1 Discrepancy", fontsize=11)
    plt.xticks(k_vals)
    plt.grid(True, axis='both', alpha=0.3, ls='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
def plot_nbm_convergence(name, grid, emp, order_pmfs, l1_results, save_path):
    """nbm 차수별 수렴 그래프"""
    plt.figure(figsize=(11, 7))
    plt.bar(grid, emp, width=1.0, alpha=0.15, color='gray', edgecolor="#333333", label='Empirical Data')
    
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(order_pmfs)))
    for i, (K, pmf) in enumerate(order_pmfs.items()):
        l1_val = l1_results.filter(pl.col('model') == f"NBM-Order {K}")['L1_diff'][0]
        label = "NB (Base)" if K == 0 else f"NBM Order {K}"
        plt.plot(grid, pmf, label=f"{label} (L1: {l1_val:.4f})", 
                 color=colors[i], ls='-' if K == 8 else '--', lw=3 if K == 8 else 1.8)
        
    plt.title(f"NBM Expansion Convergence: {name}", fontsize=15, fontweight='bold')
    plt.legend(); plt.grid(axis='y', alpha=0.2); plt.tight_layout()
    plt.savefig(save_path, dpi=300); plt.close()

def plot_comparison(name, grid, emp, model_dict, save_path):
    """
    표준 모델 비교 그래프 (개선 버전)
    - Sepsis 데이터: Poisson 및 PC PMF 제외
    - 시각적 개선: 지터(Jitter), 마커 밀도 조절, 그리기 순서 최적화 적용
    """
    plt.figure(figsize=(12, 7))

    # 1. 실제 데이터 분포 (막대 그래프)
    # 선들이 더 잘 보이도록 막대의 투명도를 높이고 색상을 중립적인 회색톤으로 설정
    plt.bar(grid, emp, width=0.8, alpha=0.15, label="Empirical", color="#2c3e50", zorder=1)

    # 2. 모델별 스타일 및 지터(Jitter) 설정
    # 각 모델의 마커와 선이 겹치지 않도록 x축에 미세한 offset(지터)을 부여
    styles = {
        "Poisson": {"color": "#95a5a6", "ls": "--", "lw": 1.5, "offset": -0.3, "marker": "x", "z": 2},
        "NB":      {"color": "#e67e22", "ls": "--", "lw": 1.5, "offset": -0.1, "marker": "s", "z": 2},
        "PC":      {"color": "#2980b9", "ls": "-",  "lw": 2.0, "offset": 0.1,  "marker": "^", "z": 3},
        "NBM":     {"color": "#c0392b", "ls": "-",  "lw": 2.5, "offset": 0.3,  "marker": "o", "z": 4}
    }

    # 3. 데이터가 너무 많을 경우 마커가 뭉쳐 보이는 문제 해결 (Sepsis 등 고빈도 데이터용)
    # 데이터 포인트가 60개를 넘어가면 마커를 5개 간격으로 표시
    m_every = 1
    if len(grid) > 60:
        m_every = 5

    # 4. Sepsis 데이터 예외 처리: Poisson과 PC 제외
    excluded_models = []
    if "Sepsis" in name:
        excluded_models = ["Poisson", "PC"]

    # 5. 모델별 PMF 플로팅
    for label, st in styles.items():
        if label in model_dict and label not in excluded_models:
            # x축 지터 적용
            jittered_grid = grid + st["offset"]
            
            plt.plot(jittered_grid, model_dict[label],
                     label=label,
                     color=st["color"],
                     ls=st["ls"],
                     lw=st["lw"],
                     marker=st["marker"],
                     markersize=5,
                     markevery=m_every, # 마커 밀도 조절
                     alpha=0.8,         # 약간의 투명도로 겹침 부위 식별 가능하게 함
                     zorder=st["z"])    # 중요 모델(NBM)을 가장 위에 그림

    # 6. 그래프 세부 설정 및 가독성 개선
    plt.title(f"Model Fit Comparison: {name}", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Count Value", fontsize=12)
    plt.ylabel("Probability Mass", fontsize=12)

    # X축 범위 최적화: 확률값이 너무 낮은(1e-4 미만) 꼬리 부분은 잘라내어 시인성 확보
    #relevant_idx = np.where(emp > 1e-4)[0]
    #if len(relevant_idx) > 0:
    #    plt.xlim(max(0, grid[relevant_idx[0]] - 2), grid[relevant_idx[-1]] + 5)

    plt.legend(frameon=True, shadow=True, loc='upper right', fontsize=11)
    plt.grid(axis='both', alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()