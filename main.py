# main.py
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from src.data_prep import load_fifa_counts, insurance_bimodal_to_count
from src.baselines import poisson_baseline, nb_pmf_mom_or_poisson, xmax_from_data
from src.expansions_mom import fit_pc_pmf, fit_meixner_pmf, normalize_pmf
from src.metrics import empirical_pmf, l1_sum_abs
from src.analysis import run_convergence_study, plot_comparison, plot_pc_convergence

RESULT_DIR = "result"
os.makedirs(RESULT_DIR, exist_ok=True)

# --- [시뮬레이션 데이터 생성기] ---
def get_simul_heavy():
    np.random.seed(129)
    return np.concatenate([stats.nbinom.rvs(15, 15/(15+20), size=1600), stats.nbinom.rvs(10, 10/(10+80), size=2400)])

def get_simul_success():
    np.random.seed(42)
    return np.concatenate([np.random.poisson(3, 5000), np.random.poisson(8, 5000)])

def write_report_section(f, title, data_stats, table_data):
    """지정된 형식으로 리포트 섹션 작성"""
    f.write("="*125 + "\n")
    f.write(f"{title:^125}\n")
    f.write("="*125 + "\n\n")
    f.write(f"▶ Dataset Stats: n={data_stats['n']}, Mean={data_stats['mu']:.4f}, V/M={data_stats['vm']:.4f}\n\n")
    
    f.write("  [ 1. L1 Discrepancy ]\n")
    f.write("  " + "-"*50 + "\n")
    for _, row in table_data.iterrows():
        f.write(f"  {row['model']:<25} | {row['L1_diff']:>18.6f}\n")
    f.write("  " + "-"*50 + "\n\n")
    
    f.write("  [ 2. Parameter Details ]\n")
    header = f"  {'Model':<15} | {'mu':>8} | {'t1':>8} | {'t2':>8} | {'t3':>8} | {'t4':>8} | {'t5':>8} | {'t6':>8} | {'t7':>8} | {'t8':>8} |"
    f.write(header + "\n")
    f.write("  " + "-" * (len(header) - 2) + "\n")
    
    for _, row in table_data.iterrows():
        p = row['params']
        mu_val = p.get('mu', data_stats['mu'])
        t_list = p.get('theta', [0.0])
        t_display = [0.0] * 8
        for i, val in enumerate(t_list[1:]):
            if i < 8: t_display[i] = val
        line = f"  {row['model']:<15} | {mu_val:>8.3f} |"
        for t_val in t_display:
            line += f" {t_val:>8.4f} |"
        f.write(line + "\n")
    f.write("\n\n")

def main():
    # 데이터 로드 및 생성
    fifa = load_fifa_counts("data/results.csv")
    ins, _ = insurance_bimodal_to_count("data/insurance.csv", bin_width=3000)
    simul_heavy = get_simul_heavy()
    simul_success = get_simul_success()

    # 분석 그룹 분리 
    standard_tasks = [("FIFA", fifa), ("Insurance", ins), ("Simul_Heavy", simul_heavy)]
    convergence_tasks = [("Insurance", ins), ("Simul_Success", simul_success)]

    with open(f"{RESULT_DIR}/unified_report.txt", "w", encoding="utf-8") as f:
        # 1부: STANDARD COMPARISON (FIFA -> Insurance -> Simul_Heavy)
        for name, data in standard_tasks:
            print(f"📊 [Standard] 분석 중: {name}...")
            mu, vm = np.mean(data), np.var(data)/np.mean(data)
            stats_dict = {'n': len(data), 'mu': mu, 'vm': vm}
            grid = np.arange(int(np.percentile(data, 99.7)) + 1)
            emp = (empirical_pmf(data, grid[-1])[:len(grid)])
            emp /= emp.sum()

            p_nbm, i_nbm = fit_meixner_pmf(data, grid)
            p_pc, i_pc = fit_pc_pmf(data, grid)
            p_nb, i_nb = nb_pmf_mom_or_poisson(data, grid)
            p_pois = stats.poisson.pmf(grid, mu)

            m_dict = {"NBM": normalize_pmf(p_nbm), "NB": normalize_pmf(p_nb), "PC": normalize_pmf(p_pc), "Poisson": normalize_pmf(p_pois)}
            plot_comparison(name, grid, emp, m_dict, f"{RESULT_DIR}/plot_{name}_std.png")
            
            std_rows = [
                {"model": "NBM", "L1_diff": l1_sum_abs(emp, m_dict["NBM"]), "params": i_nbm},
                {"model": "NB", "L1_diff": l1_sum_abs(emp, m_dict["NB"]), "params": i_nb},
                {"model": "PC", "L1_diff": l1_sum_abs(emp, m_dict["PC"]), "params": i_pc},
                {"model": "Poisson", "L1_diff": l1_sum_abs(emp, m_dict["Poisson"]), "params": {"mu": mu, "theta": [0]}}
            ]
            write_report_section(f, f"{name}: STANDARD MODEL COMPARISON", stats_dict, pd.DataFrame(std_rows))

        # 2부: CONVERGENCE STUDY (Insurance -> Simul_Success)
        for name, data in convergence_tasks:
            print(f"📈 [Convergence] 분석 중: {name}...")
            mu, vm = np.mean(data), np.var(data)/np.mean(data)
            stats_dict = {'n': len(data), 'mu': mu, 'vm': vm}
            grid = np.arange(int(np.percentile(data, 99.7)) + 1)
            emp = (empirical_pmf(data, grid[-1])[:len(grid)])
            emp /= emp.sum()

            df_conv, p_conv = run_convergence_study(data, grid, emp, mu, orders=[0, 2, 4, 6, 8])
            plot_pc_convergence(name, grid, emp, p_conv, df_conv, f"{RESULT_DIR}/plot_{name}_convergence.png")
            
            conv_rows = [{"model": r['model'], "L1_diff": r['L1_diff'], "params": {"mu": mu, "theta": r['theta']}} for _, r in df_conv.iterrows()]
            write_report_section(f, f"{name}: PC EXPANSION CONVERGENCE REPORT", stats_dict, pd.DataFrame(conv_rows))

    print(f"✅ 리포트 순서 조정 완료! (Standard 3개 -> Convergence 2개)")

if __name__ == "__main__":
    main()