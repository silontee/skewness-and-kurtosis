# main_simul.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from src.baselines import poisson_baseline, nb_pmf_mom_or_poisson, xmax_from_data
from src.expansions_mom import fit_pc_pmf, fit_meixner_pmf
from src.metrics import empirical_pmf, l1_sum_abs

RESULT_DIR = "result"
def ensure_dirs(): os.makedirs(RESULT_DIR, exist_ok=True)

def pmf_to_safe(p):
    p = np.asarray(p, dtype=float)
    p[p < 0] = 0.0
    s = p.sum()
    return p / s if s > 0 else np.ones_like(p) / len(p)

# --- 시뮬레이션 데이터 생성 (Peak 20, 80) ---
def generate_final_count_data(n_samples=4000, seed=129):
    np.random.seed(seed)
    n1 = int(n_samples * 0.4)
    n2 = n_samples - n1
    g1 = stats.nbinom.rvs(n=15, p=15/(15+20), size=n1)
    g2 = stats.nbinom.rvs(n=10, p=10/(10+80), size=n2) 
    return np.concatenate([g1, g2])

# --- 시각화: 표준 비교 (Poisson/NB 점선, Jitter 적용) ---
def plot_standard(name, grid, emp, model_dict, save_path):
    plt.figure(figsize=(11, 7))
    # Empirical: 외각선(edgecolor) 추가
    plt.bar(grid, emp, width=1.0, alpha=0.3, label="Empirical (Simul)", color="#27ae60", edgecolor="black", linewidth=0.5)
    
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
            
    plt.title(f"Standard Model Comparison (Simulation): {name}", fontsize=15, fontweight='bold')
    plt.xlabel("Count (x)"); plt.ylabel("Probability")
    plt.legend(); plt.grid(axis='y', alpha=0.3, linestyle=':'); plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()

# --- 리포트 작성 (데이터 정보 + 4대 모델 결과) ---
def write_simulation_report(path, tables, summaries, data):
    with open(path, "w", encoding="utf-8") as f:
        f.write("="*125 + "\n")
        f.write(f"{'SNP SIMULATION ANALYSIS REPORT (4-MODEL FOCUS)':^125}\n")
        f.write("="*125 + "\n\n")
        
        f.write("▶ [ DATASET INFORMATION ]\n")
        f.write(f"  - Sample Size (n): {len(data):,}\n")
        f.write(f"  - Mean: {np.mean(data):.4f} | Variance: {np.var(data):.4f}\n")
        f.write(f"  - Overdispersion (V/M): {np.var(data)/np.mean(data):.4f}\n")
        f.write(f"  - Bimodal Design: Peak 1 (mu=20) | Peak 2 (mu=80)\n")
        f.write(f"  - Actual Data Max: {int(np.max(data))}\n\n")

        for (title, table), summary in zip(tables, summaries):
            f.write(f"▶ {title}\n")
            f.write("  [ 1. L1 Discrepancy Ranking ]\n")
            f.write("  " + "-"*55 + "\n")
            f.write(f"  {'Model':<25} | {'L1 Diff (Error)':>25}\n")
            f.write("  " + "-"*55 + "\n")
            for _, r in table.iterrows():
                f.write(f"  {r['model']:<25} | {r['L1_diff']:>25.6f}\n")
            f.write("  " + "-"*55 + "\n\n")
            
            f.write("  [ 2. Parameter Details ]\n")
            header = f"  {'Model':<20} | {'mu':>7} | {'beta':>7} | {'t1':>8} | {'t2':>8} | {'t3':>8} | {'t4':>8} |"
            f.write(header + "\n" + "  " + "-" * len(header) + "\n")
            for _, r in table.iterrows():
                p = r['params']
                mu_val = float(p.get('mu', 0)) if 'mu' in p else 0.0
                beta_val = float(p.get('beta', 0)) if 'beta' in p else 0.0
                t_list = p.get('theta', [0]*5)
                
                line = f"  {r['model']:<20} | {mu_val:>7.2f} | {beta_val:>7.2f} |"
                for i in range(1, 5):
                    val = t_list[i] if i < len(t_list) else 0.0
                    line += f" {val:>8.4f} |"
                f.write(line + "\n")
            f.write("\n" + "."*125 + "\n\n")

def main():
    ensure_dirs()
    data = generate_final_count_data()
    
    # 1. x축 범위 설정 및 Empirical PMF 계산 (99.7% 분위수)
    xmax = int(np.percentile(data, 99.7)) 
    grid = np.arange(xmax + 1)
    emp_full = empirical_pmf(data, int(np.max(data)))
    emp = emp_full[:len(grid)]
    emp = emp / emp.sum() # 정규화
    
    tables, summaries = [], []
    mu = np.mean(data)

    # 2. 4가지 주력 모델 피팅
    # fit_meixner_pmf와 fit_pc_pmf는 데이터 전체를 보고 theta를 계산함
    p_nbm, i_nbm = fit_meixner_pmf(data, grid)
    p_nb, i_nb = nb_pmf_mom_or_poisson(data, grid)
    p_pc, i_pc = fit_pc_pmf(data, grid)
    p_pois = stats.poisson.pmf(grid, mu)

    m_dict = {
        "NBM": pmf_to_safe(p_nbm),
        "NB":  pmf_to_safe(p_nb),
        "PC":  pmf_to_safe(p_pc),
        "Poisson": pmf_to_safe(p_pois)
    }

    # 3. 결과 테이블 구성 (NBM -> NB -> PC -> Poisson 순서)
    order = ["NBM", "NB", "PC", "Poisson"]
    rows = []
    for k in order:
        p = m_dict[k]
        inf = i_nbm if k == "NBM" else i_nb if k == "NB" else i_pc if k == "PC" else {"mu": mu}
        rows.append({"model": k, "L1_diff": l1_sum_abs(emp, p), "params": inf})
    
    t_std = pd.DataFrame(rows)
    tables.append(("Simulation Standard Comparison", t_std))
    summaries.append({"n": len(data), "mu": mu})
    
    # 4. 시각화 및 리포트 저장
    plot_standard("Simulation", grid, emp, m_dict, "result/plot_simul_standard.png")
    write_simulation_report("result/report_simul.txt", tables, summaries, data)
    
    print("✅ 시뮬레이션 분석 완료! 4대 모델 비교 리포트와 그래프가 생성되었습니다.")

if __name__ == "__main__": main()