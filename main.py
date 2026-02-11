# main.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from src.data_prep import load_fifa_counts, insurance_bimodal_to_count
from src.baselines import poisson_baseline, nb_pmf_mom_or_poisson, xmax_from_data
from src.expansions_mom import fit_pc_pmf, fit_meixner_pmf
from src.metrics import empirical_pmf, l1_sum_abs
from src.orthopoly import get_charlier_psi

DATA_DIR, RESULT_DIR = "data", "result"

def ensure_dirs(): os.makedirs(RESULT_DIR, exist_ok=True)

def pmf_to_safe(p):
    p = np.asarray(p, dtype=float)
    p[p < 0] = 0.0
    s = p.sum()
    return p / s if s > 0 else np.ones_like(p) / len(p)

# --- PC 차수별 수렴 분석 (0, 4, 6, 8차 위주) ---
def evaluate_pc_order_series(data, grid, emp, mu, orders=[0, 4, 6, 8]):
    base_pois = stats.poisson.pmf(grid, mu)
    psi_full = get_charlier_psi(grid, mu, K=max(orders))
    theta_all = np.mean(psi_full[data], axis=0)
    
    order_rows = []
    order_pmfs = {}
    for K in orders:
        tilt = 1.0
        for k in range(1, K + 1):
            tilt += theta_all[k] * psi_full[:, k]
        p_k = pmf_to_safe(base_pois * tilt)
        l1 = l1_sum_abs(emp, p_k)
        order_rows.append({"model": f"PC-Order {K}", "L1_diff": l1, "params": {"mu": mu, "theta": theta_all[:K+1]}})
        order_pmfs[K] = p_k
    return pd.DataFrame(order_rows), order_pmfs

# --- 시각화 1: 표준 모델 비교 (Jitter + Dash 적용) ---
def plot_standard(name, grid, emp, model_dict, save_path):
    plt.figure(figsize=(11, 7))
    # Empirical: 외곽선 추가
    plt.bar(grid, emp, width=1.0, alpha=0.3, label="Empirical", color="#2c3e50", edgecolor="black", linewidth=0.7)
    
    # 모델별 스타일 (Poisson/NB는 점선, 나머지는 실선)
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
    plt.legend(); plt.grid(axis='y', alpha=0.3, linestyle=':'); plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()

# --- 시각화 2: PC 수렴 분석 (0, 4, 6, 8차) ---
def plot_pc_convergence(name, grid, emp, order_pmfs, save_path):
    plt.figure(figsize=(11, 7))
    plt.bar(grid, emp, width=1.0, alpha=0.15, color='gray', edgecolor="#333333", label='Empirical Data')
    
    # 8차만 실선, 나머지는 점선
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(order_pmfs)))
    for i, (K, pmf) in enumerate(order_pmfs.items()):
        ls = '-' if K == 8 else '--'
        alpha = 1.0 if K == 8 else 0.7
        label = "Poisson (Base)" if K == 0 else f"PC Order {K}"
        plt.plot(grid, pmf, label=label, color=colors[i], ls=ls, lw=2.5 if K==8 else 1.8, alpha=alpha)
        
    plt.title(f"PC Expansion Convergence: {name} (Order 4 to 8)", fontsize=15, fontweight='bold')
    plt.xlabel("Count (x)"); plt.ylabel("Probability")
    plt.legend(); plt.grid(axis='y', alpha=0.2); plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()

# --- 리포트 작성 (생략된 파라미터 출력부 포함) ---
def write_final_report(path, tables, summaries):
    with open(path, "w", encoding="utf-8") as f:
        f.write("="*125 + "\n")
        f.write(f"{'SNP COMPREHENSIVE ANALYSIS: SHAPE RECOVERY & ACCURACY':^125}\n")
        f.write("="*125 + "\n\n")
        for (title, table), summary in zip(tables, summaries):
            f.write(f"▶ {title}\n")
            f.write(f"  Stats: n={summary['n']}, Mean={summary['mu']:.4f}\n\n")
            f.write("  [ 1. L1 Discrepancy Table ]\n")
            f.write("  " + "-"*50 + "\n")
            for _, r in table.iterrows():
                f.write(f"  {r['model']:<25} | {r['L1_diff']:>18.6f}\n")
            f.write("  " + "-"*50 + "\n\n")
            f.write("  [ 2. Parameter Details (Theta Expansion) ]\n")
            max_t = max([len(r['params'].get('theta', [])) for _, r in table.iterrows()]) - 1
            header = f"  {'Model':<20} | {'mu':>7} |"
            for i in range(1, max_t + 1): header += f" {'t'+str(i):>8} |"
            f.write(header + "\n" + "  " + "-" * len(header) + "\n")
            for _, r in table.iterrows():
                p = r['params']
                line = f"  {r['model']:<20} | {float(p.get('mu',0)):>7.3f} |"
                t_list = p.get('theta', [])
                for i in range(1, max_t + 1):
                    val = t_list[i] if i < len(t_list) else 0.0
                    line += f" {val:>8.4f} |"
                f.write(line + "\n")
            f.write("\n" + "."*125 + "\n\n")

def main():
    ensure_dirs()
    fifa_csv, ins_csv = "data/results.csv", "data/insurance.csv"
    fifa = load_fifa_counts(fifa_csv)
    ins, _ = insurance_bimodal_to_count(ins_csv, bin_width=3000, cap_p=95)
    
    tables, summaries = [], []

    # 1. FIFA Standard
    grid_f = np.arange(xmax_from_data(fifa, q=0.999) + 1)
    emp_f = empirical_pmf(fifa, len(grid_f)-1)
    mu_f, _ = poisson_baseline(fifa)
    m_f = {"NBM": pmf_to_safe(fit_meixner_pmf(fifa, grid_f)[0]), 
           "NB": pmf_to_safe(nb_pmf_mom_or_poisson(fifa, grid_f)[0]),
           "PC": pmf_to_safe(fit_pc_pmf(fifa, grid_f)[0]), 
           "Poisson": pmf_to_safe(stats.poisson.pmf(grid_f, mu_f))}
    t_f = pd.DataFrame([{"model": k, "L1_diff": l1_sum_abs(emp_f, v), "params": {"mu": mu_f, "theta": [0]*5}} for k, v in m_f.items()])
    tables.append(("FIFA Standard Analysis", t_f)); summaries.append({"n": len(fifa), "mu": mu_f})
    plot_standard("FIFA", grid_f, emp_f, m_f, "result/plot_fifa_standard.png")

    # 2. Insurance Standard
    grid_i = np.arange(xmax_from_data(ins, q=0.999) + 1)
    emp_i = empirical_pmf(ins, len(grid_i)-1)
    mu_i, _ = poisson_baseline(ins)
    m_i = {"NBM": pmf_to_safe(fit_meixner_pmf(ins, grid_i)[0]), 
           "NB": pmf_to_safe(nb_pmf_mom_or_poisson(ins, grid_i)[0]),
           "PC": pmf_to_safe(fit_pc_pmf(ins, grid_i)[0]), 
           "Poisson": pmf_to_safe(stats.poisson.pmf(grid_i, mu_i))}
    t_i = pd.DataFrame([{"model": k, "L1_diff": l1_sum_abs(emp_i, v), "params": {"mu": mu_i, "theta": [0]*5}} for k, v in m_i.items()])
    tables.append(("Insurance Standard Analysis", t_i)); summaries.append({"n": len(ins), "mu": mu_i})
    plot_standard("Insurance", grid_i, emp_i, m_i, "result/plot_ins_standard.png")

    # 3. Insurance PC Orders (0, 4, 6, 8)
    t_pc_ord, p_pc_ord = evaluate_pc_order_series(ins, grid_i, emp_i, mu_i, orders=[0, 4, 6, 8])
    tables.append(("Insurance PC Order Convergence Study", t_pc_ord)); summaries.append({"n": len(ins), "mu": mu_i})
    plot_pc_convergence("Insurance", grid_i, emp_i, p_pc_ord, "result/plot_ins_pc_convergence.png")

    write_final_report("result/report.txt", tables, summaries)
    print("✅ 실제 데이터 분석 완료! 8차 theta 값이 포함된 리포트와 고도화된 그래프 3개가 생성되었습니다.")

if __name__ == "__main__": main()