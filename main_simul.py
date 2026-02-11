import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# 프로젝트 내부 모듈 적극 활용
from src.baselines import nb_pmf_mom_or_poisson
from src.expansions_mom import fit_pc_pmf, fit_meixner_pmf, normalize_pmf #
from src.metrics import empirical_pmf, l1_sum_abs #

RESULT_DIR = "result"
os.makedirs(RESULT_DIR, exist_ok=True)

def generate_final_count_data(n_samples=4000, seed=129):
    """Peak 20, 80을 가진 과분산 이봉성 시뮬레이션 데이터 생성"""
    np.random.seed(seed)
    n1, n2 = int(n_samples * 0.4), n_samples - int(n_samples * 0.4)
    # NB 분포를 섞어 고의적으로 과분산(V/M > 20) 상황 유도
    g1 = stats.nbinom.rvs(n=15, p=15/(15+20), size=n1)
    g2 = stats.nbinom.rvs(n=10, p=10/(10+80), size=n2) 
    return np.concatenate([g1, g2])

def plot_standard(name, grid, emp, model_dict, save_path):
    """표준 모델 비교 시각화 (Poisson/NB vs PC/NBM)"""
    plt.figure(figsize=(11, 7))
    # grid와 emp의 길이가 일치해야 함
    plt.bar(grid, emp, width=1.0, alpha=0.3, label="Empirical (Simul)", color="#27ae60", edgecolor="black")
    
    styles = {
        "Poisson": {"color": "#95a5a6", "ls": "--", "lw": 1.5, "offset": -0.4},
        "NB":      {"color": "#e67e22", "ls": "--", "lw": 1.5, "offset": -0.15},
        "PC":      {"color": "#2980b9", "ls": "-",  "lw": 2.2, "offset": 0.15},
        "NBM":     {"color": "#c0392b", "ls": "-",  "lw": 2.2, "offset": 0.4}
    }
    for label, st in styles.items():
        if label in model_dict:
            plt.plot(grid + st["offset"], model_dict[label], label=label, 
                     color=st["color"], ls=st["ls"], lw=st["lw"], marker='o', ms=4)
            
    plt.title(f"Simulation Analysis: NBM vs Others (V/M > 20)", fontsize=15, fontweight='bold')
    plt.xlabel("Count (x)"); plt.ylabel("Probability")
    plt.legend(); plt.grid(axis='y', alpha=0.3, linestyle=':'); plt.tight_layout()
    plt.savefig(save_path, dpi=300); plt.close()

def main():
    data = generate_final_count_data()
    mu = np.mean(data)
    
    # 1. 그리드 설정 및 실제 분포 계산 (에러 수정 포인트)
    grid = np.arange(int(np.percentile(data, 99.7)) + 1)
    # empirical_pmf가 데이터의 실제 최대값까지 반환할 수 있으므로 grid 길이에 맞춰 슬라이싱
    emp_full = empirical_pmf(data, grid[-1])
    emp = emp_full[:len(grid)] 
    emp = emp / emp.sum() # 슬라이싱 후 확률 합을 1로 재정규화
    
    # 2. 4대 모델 피팅
    p_nbm, i_nbm = fit_meixner_pmf(data, grid)
    p_nb, i_nb   = nb_pmf_mom_or_poisson(data, grid)
    p_pc, i_pc   = fit_pc_pmf(data, grid)
    p_pois       = stats.poisson.pmf(grid, mu)

    # 3. 정규화 및 결과 묶기
    m_dict = {
        "NBM": normalize_pmf(p_nbm), "NB": normalize_pmf(p_nb),
        "PC":  normalize_pmf(p_pc),  "Poisson": normalize_pmf(p_pois)
    }

    # 4. 성능 비교 테이블 생성
    order = ["NBM", "NB", "PC", "Poisson"]
    rows = []
    for k in order:
        inf = i_nbm if k == "NBM" else i_nb if k == "NB" else i_pc if k == "PC" else {"mu": mu}
        rows.append({"model": k, "L1_diff": l1_sum_abs(emp, m_dict[k]), "params": inf})
    
    # 5. 시각화 및 리포트 출력
    plot_standard("Final_Simulation", grid, emp, m_dict, f"{RESULT_DIR}/plot_simul_standard.png")
    
    print(f"\n{'='*40}\n{'SIMULATION RESULT SUMMARY':^40}\n{'='*40}")
    print(f"{'Model':<15} | {'L1 Error':>15}")
    print("-" * 33)
    for r in rows:
        print(f"{r['model']:<15} | {r['L1_diff']:>15.6f}")
    print('='*40)

if __name__ == "__main__":
    main()