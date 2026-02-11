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

# --- PC 차수별 계산 (0, 1, 2, 3, 4, 8차) ---
def evaluate_pc_order_series(data, grid, emp, mu, orders=[0, 1, 2, 3, 4, 8]):
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
        # 리포트 출력을 위해 params에 mu와 8차까지의 theta를 다 넣어둡니다.
        order_rows.append({
            "model": f"PC-Order {K}", 
            "L1_diff": l1, 
            "params": {"mu": mu, "theta": theta_all[:K+1]}
        })
        order_pmfs[K] = p_k
    return pd.DataFrame(order_rows), order_pmfs

# --- 시각화 함수들 ---
def plot_standard(name, grid, emp, model_dict, save_path):
    plt.figure(figsize=(10, 6))
    plt.bar(grid, emp, width=1.0, alpha=0.3, label="Empirical", color="#2c3e50")
    styles = {"Poisson": "#95a5a6", "NB": "#e67e22", "PC": "#2980b9", "NBM": "#c0392b"}
    for label, pmf in model_dict.items():
        plt.plot(grid, pmf, label=label, color=styles[label], lw=2, marker='o', ms=3)
    plt.title(f"Standard Comparison: {name}"); plt.legend(); plt.savefig(save_path); plt.close()

def plot_pc_limit(name, grid, emp, order_pmfs, save_path):
    plt.figure(figsize=(10, 6))
    plt.bar(grid, emp, width=1.0, alpha=0.2, color='gray', label='Empirical')
    colors = plt.cm.viridis(np.linspace(0, 1, len(order_pmfs)))
    for i, (K, pmf) in enumerate(order_pmfs.items()):
        plt.plot(grid, pmf, label=f"PC Order {K}" if K>0 else "Poisson", color=colors[i], lw=2)
    plt.title(f"PC Expansion Convergence Limit: {name}"); plt.legend(); plt.savefig(save_path); plt.close()

# --- 리포트 작성 (theta 8차 지원) ---
def write_final_report(path, tables, summaries):
    with open(path, "w", encoding="utf-8") as f:
        f.write("="*120 + "\n")
        f.write(f"{'SNP COMPREHENSIVE ANALYSIS REPORT':^120}\n")
        f.write("="*120 + "\n\n")

        for (title, table), summary in zip(tables, summaries):
            f.write(f"▶ {title}\n")
            f.write(f"  Stats: n={summary['n']}, Mean={summary['mu']:.4f}\n\n")
            
            # 1. L1 Table
            f.write("  [ 1. L1 Discrepancy ]\n")
            f.write("  " + "-"*50 + "\n")
            for _, r in table.iterrows():
                f.write(f"  {r['model']:<25} | {r['L1_diff']:>18.6f}\n")
            f.write("  " + "-"*50 + "\n\n")

            # 2. Parameter Table (Dynamic Theta)
            f.write("  [ 2. Parameter Details ]\n")
            # 최대 차수를 계산하여 헤더 생성
            max_t = 0
            for _, r in table.iterrows():
                t_len = len(r['params'].get('theta', [])) - 1
                if t_len > max_t: max_t = t_len
            
            header = f"  {'Model':<20} | {'mu':>7} | {'beta':>7} |"
            for i in range(1, max_t + 1): header += f" {'t'+str(i):>8} |"
            f.write(header + "\n")
            f.write("  " + "-" * len(header) + "\n")

            for _, r in table.iterrows():
                p = r['params']
                mu = f"{float(p.get('mu', 0)):.3f}" if 'mu' in p else "-"
                beta = f"{float(p.get('beta', 0)):.3f}" if 'beta' in p else "-"
                line = f"  {r['model']:<20} | {mu:>7} | {beta:>7} |"
                
                t_list = p.get('theta', [])
                for i in range(1, max_t + 1):
                    val = t_list[i] if i < len(t_list) else 0.0
                    line += f" {val:>8.4f} |"
                f.write(line + "\n")
            f.write("\n" + "."*120 + "\n\n")

def main():
    ensure_dirs()
    fifa = load_fifa_counts("data/results.csv")
    ins, _ = insurance_bimodal_to_count("data/insurance.csv", bin_width=3000, cap_p=95)

    tables, summaries = [], []

    # (1) FIFA Standard
    grid_f = np.arange(xmax_from_data(fifa, q=0.999) + 1)
    emp_f = empirical_pmf(fifa, len(grid_f)-1)
    mu_f, _ = poisson_baseline(fifa)
    p_nbm_f, i_nbm_f = fit_meixner_pmf(fifa, grid_f)
    p_nb_f, i_nb_f = nb_pmf_mom_or_poisson(fifa, grid_f)
    p_pc_f, i_pc_f = fit_pc_pmf(fifa, grid_f)
    
    m_f = {"NBM": pmf_to_safe(p_nbm_f), "NB": pmf_to_safe(p_nb_f), "PC": pmf_to_safe(p_pc_f), "Poisson": pmf_to_safe(stats.poisson.pmf(grid_f, mu_f))}
    t_f = pd.DataFrame([{"model": k, "L1_diff": l1_sum_abs(emp_f, v), "params": (i_nbm_f if k=="NBM" else i_nb_f if k=="NB" else i_pc_f if k=="PC" else {"mu": mu_f})} for k, v in m_f.items()])
    tables.append(("FIFA Standard Comparison", t_f))
    summaries.append({"n": len(fifa), "mu": mu_f})
    plot_standard("FIFA", grid_f, emp_f, m_f, "result/plot_fifa_standard.png")

    # (2) Insurance Standard
    grid_i = np.arange(xmax_from_data(ins, q=0.999) + 1)
    emp_i = empirical_pmf(ins, len(grid_i)-1)
    mu_i, _ = poisson_baseline(ins)
    p_nbm_i, i_nbm_i = fit_meixner_pmf(ins, grid_i)
    p_nb_i, i_nb_i = nb_pmf_mom_or_poisson(ins, grid_i)
    p_pc_i, i_pc_i = fit_pc_pmf(ins, grid_i)

    m_i = {"NBM": pmf_to_safe(p_nbm_i), "NB": pmf_to_safe(p_nb_i), "PC": pmf_to_safe(p_pc_i), "Poisson": pmf_to_safe(stats.poisson.pmf(grid_i, mu_i))}
    t_i = pd.DataFrame([{"model": k, "L1_diff": l1_sum_abs(emp_i, v), "params": (i_nbm_i if k=="NBM" else i_nb_i if k=="NB" else i_pc_i if k=="PC" else {"mu": mu_i})} for k, v in m_i.items()])
    tables.append(("Insurance Standard Comparison", t_i))
    summaries.append({"n": len(ins), "mu": mu_i})
    plot_standard("Insurance", grid_i, emp_i, m_i, "result/plot_ins_standard.png")

    # (3) Insurance PC Orders (Table 3 & Plot 3)
    t_pc_ord, p_pc_ord = evaluate_pc_order_series(ins, grid_i, emp_i, mu_i)
    tables.append(("Insurance PC Order Convergence Analysis", t_pc_ord))
    summaries.append({"n": len(ins), "mu": mu_i})
    plot_pc_limit("Insurance", grid_i, emp_i, p_pc_ord, "result/plot_ins_pc_limit.png")

    write_final_report("result/report.txt", tables, summaries)
    print("✅ 분석 완료! 표 3개와 그래프 3개가 result 폴더에 생성되었습니다.")

if __name__ == "__main__": main()