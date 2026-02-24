# main.py
import os
import numpy as np
import scipy.stats as stats
from src.data_prep_v2 import load_fifa_counts, insurance_bimodal_to_count, load_sepsis_lab_counts
from src.baselines import nb_pmf_mom_or_poisson, xmax_from_data
from src.expansions_mom import fit_pc_pmf, fit_meixner_pmf, normalize_pmf
from src.metrics_v2 import calculate_data_stats, empirical_pmf, l1_sum_abs
from src.analysis_v2 import run_nbm_convergence_study, plot_nbm_l1_convergence, plot_nbm_convergence, plot_comparison

RESULT_DIR = "result_v3"
os.makedirs(RESULT_DIR, exist_ok=True)

def write_report_section(f, title, data_stats, table_data):
    """지정된 형식으로 리포트 섹션 작성"""
    f.write("="*125 + "\n")
    f.write(f"{title:^125}\n")
    f.write("="*125 + "\n\n")
    f.write(f"▶ Dataset Stats:\n")
    f.write(f"- Sample Size (n): {data_stats['n']}\n")
    f.write(f"- Mean (mu): {data_stats['mu']}\n")
    f.write(f"- Variance: {data_stats['var']}\n")
    f.write(f"- V/M Ratio: {data_stats['vm']}\n")
    f.write(f"- Heavy Tail: {data_stats['heavy_tail']} (Kurtosis: {data_stats['kurtosis']})\n\n")
    
    f.write("  [ 1. L1 Discrepancy ]\n")
    f.write("  " + "-"*50 + "\n")
    for row in table_data:
        f.write(f"  {row['model']:<25} | {row['L1_diff']:>18.6f}\n")
    f.write("  " + "-"*50 + "\n\n")

    
    f.write("  [ 2. Parameter Details ]\n")    
    # 출력할 최대 theta 차수 결정
    if "STANDARD" in title.upper():
        max_t = 4
    else:
        # 데이터 내의 theta 리스트 중 가장 긴 것을 기준으로 컬럼 생성
        max_t = 0
        for row in table_data:
            t_len = len(row.get('params', {}).get('theta', [])) - 1
            if t_len > max_t: max_t = t_len
            
    # 헤더 구성 (Model, mu/beta, c, t1...tN)
    header = f"  {'Model':<20} | {'mu':>8}"
    for i in range(1, max_t + 1):
        header += f" | {f't{i}':>8}"
    header += " |"

    f.write(header + "\n")
    f.write("  " + "-" * (len(header) - 2) + "\n")

    # 데이터 행 작성
    for row in table_data:
        p = row.get('params', {})
        mu_val = p.get('mu', data_stats['mu'])
        theta = p.get('theta', [])

        # 기본 정보 (Model, mu)
        line = f"  {row['model']:<20} | {mu_val:>8.3f}"
        
        # t1부터 max_t까지 채우기
        for i in range(1, max_t + 1):
            if i < len(theta):
                val = theta[i]
                if isinstance(val, (int, float)):
                    line += f" | {val:>8.4f}"
                else:
                    line += f" | {str(val):>8}"
            else:
                line += f" | {0.0:>8.4f}"
        line += " |"
        f.write(line + "\n")
    f.write("\n")

def main():
    # 데이터 로드 및 생성
    datasets = [
        ("FIFA", load_fifa_counts("data/results.csv")),
        ("Insurance", insurance_bimodal_to_count("data/insurance.csv")),
        ("Sepsis_Lab_Count", load_sepsis_lab_counts("data/Dataset.csv"))
    ]

    with open(f"{RESULT_DIR}/report_v3.txt", "w", encoding="utf-8") as f:
        # 1부: STANDARD COMPARISON
        for name, data in datasets:
            print(f"📊 [Standard] 분석 중: {name}...")
            stats_dict = calculate_data_stats(data)
            mu = stats_dict['mu']
            grid = np.arange(xmax_from_data(data) + 1)
            emp = (empirical_pmf(data, grid[-1])[:len(grid)])
            emp /= emp.sum()

            p_nbm, i_nbm = fit_meixner_pmf(data, grid)
            p_pc, i_pc = fit_pc_pmf(data, grid)
            p_nb, i_nb = nb_pmf_mom_or_poisson(data, grid)
            p_pois = stats.poisson.pmf(grid, mu)

            m_dict = {
                "Poisson": normalize_pmf(p_pois),
                "PC": normalize_pmf(p_pc),
                "NB": normalize_pmf(p_nb),
                "NBM": normalize_pmf(p_nbm)
            }
            
            plot_comparison(name, grid, emp, m_dict, f"{RESULT_DIR}/plot_{name}_std.png")
            
            std_rows = [
                {"model": "Poisson", "L1_diff": l1_sum_abs(emp, m_dict["Poisson"]), "params": {"mu": mu, "theta": [0]}},
                {"model": "PC", "L1_diff": l1_sum_abs(emp, m_dict["PC"]), "params": i_pc},
                {"model": "NB", "L1_diff": l1_sum_abs(emp, m_dict["NB"]), "params": i_nb},
                {"model": "NBM", "L1_diff": l1_sum_abs(emp, m_dict["NBM"]), "params": i_nbm}
            ]
            write_report_section(f, f"{name}: STANDARD MODEL COMPARISON", stats_dict, std_rows)

        # 2부: CONVERGENCE STUDY
        for name, data in datasets:
            print(f"📈 [Convergence] 분석 중: {name}...")
            stats_dict = calculate_data_stats(data)
            mu = stats_dict['mu']
            grid = np.arange(xmax_from_data(data) + 1)
            emp = (empirical_pmf(data, grid[-1])[:len(grid)])
            emp /= emp.sum()

            df_conv, p_conv = run_nbm_convergence_study(data, grid, emp)
            conv_rows = [{"model": r['model'], "L1_diff": r['L1_diff'], "params": {"beta": r['beta'], "c": r['c'], "theta": r['theta']}} for r in df_conv.iter_rows(named=True)]

            write_report_section(f, f"{name}: NBM CONVERGENCE REPORT", stats_dict, conv_rows)
            plot_nbm_l1_convergence(name, df_conv, os.path.join(RESULT_DIR, f"plot_nbm_L1_{name}.png"))
            plot_nbm_convergence(name, grid, emp, p_conv, df_conv, f"{RESULT_DIR}/plot_nbm_conv_{name}.png")
            
    print(f"✅ 분석 완료! '{RESULT_DIR}' 폴더에서 결과표를 확인하세요.")

if __name__ == "__main__":
    main()