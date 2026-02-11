import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# 프로젝트 내부 모듈 임포트
from src.data_prep import load_fifa_counts
from src.baselines import poisson_baseline, xmax_from_data
from src.metrics import empirical_pmf, l1_sum_abs
from src.orthopoly import get_charlier_psi

RESULT_DIR = "result"
DATA_FILE = "data/results.csv"

def ensure_dirs():
    os.makedirs(RESULT_DIR, exist_ok=True)

def pmf_to_safe(p):
    """음수 확률 보정 및 정규화"""
    p = np.asarray(p, dtype=float)
    p[p < 0] = 0.0
    s = p.sum()
    return p / s if s > 0 else np.ones_like(p) / len(p)

def run_fifa_convergence():
    ensure_dirs()
    
    # 1. FIFA 데이터 로드 및 기초 통계량 계산
    if not os.path.exists(DATA_FILE):
        print(f"오류: {DATA_FILE} 파일을 찾을 수 없습니다.")
        return
    
    data = load_fifa_counts(DATA_FILE)
    mu = np.mean(data)
    xmax = xmax_from_data(data, q=0.999)
    grid = np.arange(xmax + 1)
    emp = empirical_pmf(data, xmax)
    
    print(f"--- FIFA 데이터 통계 ---")
    print(f"Sample Size: {len(data)}, Mean: {mu:.4f}, V/M Ratio: {np.var(data)/mu:.4f}")
    
    # 2. 비교할 PC 차수 설정 (0차는 순수 Poisson)
    orders = [0, 4, 5, 6, 7, 8]
    
    # 차수 계산을 위한 다항식 및 계수(Theta) 사전 계산
    # IndexError 방지를 위해 데이터 전체 범위를 커버하는 다항식 생성
    full_data_grid = np.arange(np.max(data) + 1)
    psi_for_theta = get_charlier_psi(full_data_grid, mu, K=8)
    theta_all = np.mean(psi_for_theta[data], axis=0)
    
    # 시각화용 Grid 다항식
    psi_grid = get_charlier_psi(grid, mu, K=8)
    base_pois = stats.poisson.pmf(grid, mu)
    
    results_list = []
    order_pmfs = {}
    
    # 차수별 PMF 및 L1 오차 계산
    for K in orders:
        tilt = 1.0
        for k in range(1, K + 1):
            tilt += theta_all[k] * psi_grid[:, k]
        
        p_k = pmf_to_safe(base_pois * tilt)
        l1 = l1_sum_abs(emp, p_k)
        
        results_list.append({
            "Order": f"PC-Order {K}",
            "L1_diff": l1,
            "mu": mu,
            "theta": theta_all[:K+1].tolist()
        })
        order_pmfs[K] = p_k

    # 3. 시각화 (Convergence Graph)
    plt.figure(figsize=(12, 7))
    plt.bar(grid, emp, width=1.0, alpha=0.2, color='gray', edgecolor="black", label='Empirical (FIFA)')
    
    # 차수가 높아질수록 진해지는 컬러 맵 적용
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(orders)))
    
    for i, K in enumerate(orders):
        label = "Poisson (Base)" if K == 0 else f"PC Order {K}"
        # 8차만 실선으로 강조, 나머지는 점선
        ls = '-' if K == 8 else '--'
        alpha = 1.0 if K == 8 else 0.6
        lw = 2.5 if K == 8 else 1.5
        plt.plot(grid, order_pmfs[K], label=label, color=colors[i], ls=ls, lw=lw, alpha=alpha)
        
    plt.title("FIFA Convergence Test: Shape Recovery with PC Orders", fontsize=15, fontweight='bold')
    plt.xlabel("Count (Goals)"); plt.ylabel("Probability")
    plt.legend(); plt.grid(axis='y', alpha=0.3, linestyle=':')
    plt.tight_layout()
    plt.savefig("result/plot_fifa_convergence_test.png", dpi=300)
    
    # 4. 상세 리포트 생성
    report_path = "result/report_fifa_convergence.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*125 + "\n")
        f.write(f"{'FIFA DATA: PC EXPANSION CONVERGENCE REPORT':^125}\n")
        f.write("="*125 + "\n\n")
        f.write(f"▶ Dataset Stats: n={len(data)}, Mean={mu:.4f}, V/M={np.var(data)/mu:.4f}\n\n")
        
        f.write("  [ 1. L1 Discrepancy (Convergence Check) ]\n")
        f.write("  " + "-"*50 + "\n")
        for res in results_list:
            f.write(f"  {res['Order']:<25} | {res['L1_diff']:>18.6f}\n")
        f.write("  " + "-"*50 + "\n\n")
        
        f.write("  [ 2. Parameter Details ]\n")
        header = f"  {'Model':<15} | {'mu':>7} |"
        for i in range(1, 9): header += f" {'t'+str(i):>8} |"
        f.write(header + "\n" + "  " + "-" * len(header) + "\n")
        
        for res in results_list:
            mu_val = f"{res['mu']:>7.3f}"
            line = f"  {res['Order']:<15} | {mu_val} |"
            t_list = res['theta']
            for i in range(1, 9):
                val = t_list[i] if i < len(t_list) else 0.0
                line += f" {val:>8.4f} |"
            f.write(line + "\n")
            
    print(f"✅ FIFA 수렴 테스트 완료!")
    print(f"결과 저장: result/report_fifa_convergence.txt, result/plot_fifa_convergence_test.png")

if __name__ == "__main__":
    run_fifa_convergence()