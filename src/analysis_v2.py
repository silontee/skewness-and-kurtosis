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

def find_optimal_order(name, df_conv):
    """
    L1 에러가 오름차순으로 감소하다가 처음으로 상승하기 직전의 차수를 반환
    """
    l1_vals = df_conv.get_column("L1_diff").to_list()
    k_vals = [int(m.split()[-1]) for m in df_conv.get_column("model").to_list()]
    
    # [Step 1] 에러 증가 지점 찾기
    stop_idx = len(l1_vals) - 1
    for i in range(1, len(l1_vals)):
        if l1_vals[i] > l1_vals[i-1]:
            stop_idx = i - 1 # 증가하기 직전 차수
            break
            
    # [Step 2] 내림차순 역추적 (stop_idx부터 2차까지 역순)
    opt_k = k_vals[stop_idx]
    
    if "Sepsis" in name:
        # range(start, stop, step) -> stop_idx부터 2까지 역순 탐색
        for j in range(stop_idx, 1, -1):
            # 현재 차수(j)의 감소율: L1[j-1] - L1[j]
            curr_reduction = l1_vals[j-1] - l1_vals[j]
            # 이전 차수(j-1)의 감소율: L1[j-2] - L1[j-1]
            prev_reduction = l1_vals[j-2] - l1_vals[j-1]
            
            # 💡 감소율 둔화 조건: 현재 스텝의 개선 폭이 이전보다 작아지면
            if curr_reduction < prev_reduction:
                # "그 이전 차수(둔화되기 전의 효율적인 차수)"를 최적점으로 선정
                opt_k = k_vals[j-1]
                break # 최적점을 찾았으므로 루프 종료
        
    return opt_k

def plot_nbm_l1_convergence(name, df_conv, save_path):
    """최적점에 수직 점선을 포함한 L1 수렴도 그래프"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    k_vals = [int(m.split()[-1]) for m in df_conv.get_column("model")]
    l1_vals = df_conv.get_column("L1_diff").to_numpy()
    
    # 최적 차수 찾기
    opt_k = find_optimal_order(name, df_conv)
    
    ax.plot(k_vals, l1_vals, marker='o', linestyle='-', 
             color='#d35400', lw=2.5, ms=8, label='NBM L1 Error', zorder=3)
    
    # 💡 [핵심] 최적점 수직선 표시
    ax.axvline(x=opt_k, color='#c0392b', linestyle='--', lw=2, alpha=0.7, zorder=2)

    # 차수별 수치 표시
    for i, (kv, lv) in enumerate(zip(k_vals, l1_vals)):
        is_optimal = (kv == opt_k)
        
        # 최적점일 경우 스타일 변경 (빨간색, Bold)
        txt_color = 'red' if is_optimal else '#2c3e50'
        txt_weight = 'bold' if is_optimal else 'normal'
        txt_size = 11 if is_optimal else 8
        
        # 짝수 번째는 위(above), 홀수 번째는 아래(below)로 배치하여 수평 겹침 방지
        if i % 2 == 0:
            x_offset = 11
            y_offset = 15
            va = 'bottom'
        else:
            x_offset = -11
            y_offset = -15 # 아래쪽은 마커와 겹치지 않게 더 멀리 배치
            va = 'top'
        
        if is_optimal:
            if i % 2 == 0:
                x_offset = 0
                y_offset = 15
                va = 'bottom'
            else:
                x_offset = 0
                y_offset = -15 # 아래쪽은 마커와 겹치지 않게 더 멀리 배치
                va = 'top'  
        
        ax.annotate(
            f"{lv:.4f}", 
            (kv, lv), 
            textcoords="offset points", 
            xytext=(x_offset, y_offset),
            ha='center', 
            va=va, 
            fontsize=txt_size, 
            fontweight=txt_weight, 
            color=txt_color,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, ec=('red' if is_optimal else 'none'))
        )

    ax.set_title(f"L1 Convergence Trend: {name}", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Order (K)", fontsize=11)
    ax.set_ylabel("L1 Discrepancy", fontsize=11)
    ax.set_xticks(k_vals)
    ax.grid(True, alpha=0.3, ls='--')
    
    # 여백 확보
    y_min, y_max = min(l1_vals), max(l1_vals)
    ax.set_ylim(y_min - (y_max-y_min)*0.3, y_max + (y_max-y_min)*0.3)
    
    # 범례에 최적 차수 명시
    ax.legend([f'L1 Error', f'Optimal Point (K={opt_k})'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return opt_k

def plot_comparison(name, grid, emp, model_dict, save_path, opt_k=None):
    """
    기존 표준 NBM(K=4)과 최적 NBM을 동시에 표시
    """
    plt.figure(figsize=(12, 7))
    plt.bar(grid, emp, width=0.8, alpha=0.15, label="Empirical", color="#2c3e50", zorder=1)

    # 모델별 스타일 설정
    styles = {
        "Poisson":  {"color": "#95a5a6", "ls": "--", "lw": 1.5, "offset": -0.3, "marker": "x", "z": 2},
        "NB":       {"color": "#e67e22", "ls": "--", "lw": 1.5, "offset": -0.1, "marker": "s", "z": 2},
        "PC":       {"color": "#2980b9", "ls": "-",  "lw": 2.0, "offset": 0.1,  "marker": "^", "z": 3},
        "NBM":      {"color": "#c0392b", "ls": "-",  "lw": 2.2, "offset": 0.3,  "marker": "o", "z": 4},
        "NBM_Opt":  {"color": "#8e44ad", "ls": "-",  "lw": 3.0, "offset": 0.4,  "marker": "*", "z": 5}
    }

    m_every = 5 if len(grid) > 60 else 1
    # Sepsis 등에서 노이즈 제거 (FIFA는 유지)
    excluded = ["Poisson", "PC"] if "FIFA" not in name else []

    for label, st in styles.items():
        if label in model_dict and label not in excluded:
            # 레이블 이름 다듬기
            display_name = label
            if label == "NBM_Opt": display_name = f"NBM (Optimal, K={opt_k})"
            if label == "NBM": display_name = "NBM (Standard, K=4)"

            plt.plot(grid + st["offset"], model_dict[label],
                     label=display_name, color=st["color"], ls=st["ls"], lw=st["lw"],
                     marker=st["marker"], markersize=6, markevery=m_every,
                     alpha=0.9, zorder=st["z"])

    # X축 시작점 여백 확보 (Y축에 붙지 않게)
    plt.xlim(grid[0]-2, grid[-1]+2)
    
    plt.title(f"Model Fit Comparison: {name}", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Count Value"); plt.ylabel("Probability Mass")
    plt.legend(frameon=True, shadow=True, loc='upper right', fontsize=10)
    plt.grid(axis='both', alpha=0.2, ls='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300); plt.close()