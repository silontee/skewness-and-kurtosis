import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 프로젝트 내부 모듈 활용
from src.metrics import empirical_pmf, l1_sum_abs #
from src.orthopoly import get_charlier_psi #
from src.expansions_mom import normalize_pmf #

RESULT_DIR = "result"
os.makedirs(RESULT_DIR, exist_ok=True)

def generate_success_data(n_samples=10000, seed=42):
    """수렴 증명용 'Clean Bimodal' 데이터 생성"""
    np.random.seed(seed)
    n1 = n_samples // 2
    n2 = n_samples - n1
    # 평균 3과 8의 포아송 분포를 섞어 이상적인 수렴 환경 조성
    data = np.concatenate([np.random.poisson(3, n1), np.random.poisson(8, n2)])
    return data

def run_success_analysis():
    data = generate_success_data()
    mu, var = np.mean(data), np.var(data)
    
    # 1. 그리드 설정 및 실제 확률(Empirical) 계산
    grid = np.arange(int(np.max(data)) + 3)
    emp = empirical_pmf(data, grid[-1])
    
    # 2. PC 차수별 계산 (0, 2, 4, 6, 8차)
    orders = [0, 2, 4, 6, 8]
    full_psi = get_charlier_psi(np.arange(np.max(data) + 1), mu, K=max(orders)) #
    theta_all = np.mean(full_psi[data], axis=0) #
    
    psi_grid = get_charlier_psi(grid, mu, K=max(orders))
    base_pois = stats.poisson.pmf(grid, mu)
    
    l1_results = []
    order_pmfs = {}

    for K in orders:
        tilt = 1.0 + (psi_grid[:, 1:K+1] @ theta_all[1:K+1]) #
        p_k = normalize_pmf(base_pois * tilt) #
        
        l1 = l1_sum_abs(emp, p_k) #
        l1_results.append({"Order": f"PC-Order {K}", "L1": l1, "thetas": theta_all[:K+1]})
        order_pmfs[K] = p_k

    # 3. 시각화 (Convergence Graph)
    plt.figure(figsize=(12, 7))
    plt.bar(grid, emp, width=1.0, alpha=0.2, color='gray', edgecolor='black', label='Empirical')
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(orders)))
    for i, K in enumerate(orders):
        plt.plot(grid, order_pmfs[K], label=f"Order {K} (L1: {l1_results[i]['L1']:.4f})", 
                 color=colors[i], ls='-' if K == 8 else '--', lw=3 if K == 8 else 1.5)
    plt.title("Convergence Success: PC Expansion Capturing Bimodal Shape", fontsize=14, fontweight='bold')
    plt.legend(); plt.grid(axis='y', alpha=0.3); plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/plot_success_convergence.png", dpi=300)

    # 4. 리포트 작성 (FIFA 스타일 양식 적용)
    with open(f"{RESULT_DIR}/report_success_case.txt", "w", encoding="utf-8") as f:
        f.write("="*125 + "\n")
        f.write(f"{'SUCCESS CASE: PC EXPANSION CONVERGENCE REPORT':^125}\n")
        f.write("="*125 + "\n\n")
        
        f.write(f"▶ Dataset Stats: n={len(data)}, Mean={mu:.4f}, V/M={var/mu:.4f}\n\n")
        
        f.write("  [ 1. L1 Discrepancy (Convergence Check) ]\n")
        f.write("  " + "-"*50 + "\n")
        for res in l1_results:
            f.write(f"  {res['Order']:<25} | {res['L1']:>18.6f}\n")
        f.write("  " + "-"*50 + "\n\n")
        
        f.write("  [ 2. Parameter Details ]\n")
        header = f"  {'Model':<15} | {'mu':>8} | {'t1':>8} | {'t2':>8} | {'t3':>8} | {'t4':>8} | {'t5':>8} | {'t6':>8} | {'t7':>8} | {'t8':>8} |"
        f.write(header + "\n")
        f.write("  " + "-" * (len(header) - 2) + "\n")
        
        for res in l1_results:
            t_list = res['thetas']
            t_padded = [0.0] * 8 # t1 ~ t8 공간 확보
            for idx, val in enumerate(t_list[1:]): # t0 제외
                if idx < 8: t_padded[idx] = val
            
            line = f"  {res['Order']:<15} | {mu:>8.3f} |"
            for t_val in t_padded:
                line += f" {t_val:>8.4f} |"
            f.write(line + "\n")
            
    print(f"✅ 수렴 리포트가  생성되었습니다: {RESULT_DIR}/report_success_case.txt")

if __name__ == "__main__":
    run_success_analysis()