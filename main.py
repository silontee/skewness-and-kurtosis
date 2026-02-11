# main.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_prep import load_fifa_counts, insurance_bimodal_to_count
from src.baselines import poisson_baseline, nb_pmf_mom_or_poisson, xmax_from_data
from src.expansions_mom import fit_pc_pmf, fit_meixner_pmf
from src.metrics import empirical_pmf, l1_sum_abs

DATA_DIR = "data"
RESULT_DIR = "result"

def ensure_dirs():
    os.makedirs(RESULT_DIR, exist_ok=True)

def pmf_to_safe(p):
    p = np.asarray(p, dtype=float)
    # 음수 확률을 0으로 처리하는 것은 수치적 안정성을 위한 필수적 제약 조건입니다.
    p[p < 0] = 0.0
    s = p.sum()
    return p / s if s > 0 else np.ones_like(p) / len(p)

def plot_all(name, grid, emp, model_dict, save_path):
    plt.figure(figsize=(11, 7))
    # 1. Empirical: 연결된 바 형태 (데이터 밀도 강조)
    plt.bar(grid, emp, width=1.0, alpha=0.3, label="Empirical", color="#2c3e50", edgecolor="#34495e", linewidth=0.5)

    # 2. 모델별 스타일 및 Jitter 설정
    # Poisson(-0.25), NB(-0.08), PC(+0.08), NBM(+0.25)
    styles = {
        "Poisson": {"color": "#95a5a6", "ls": "--", "lw": 1.5, "offset": -0.25, "marker": "x", "ms": 5},
        "NB":      {"color": "#e67e22", "ls": "--", "lw": 1.5, "offset": -0.08, "marker": "v", "ms": 5},
        "PC":      {"color": "#2980b9", "ls": "-",  "lw": 2.2, "offset": 0.08,  "marker": "o", "ms": 6},
        "NBM":     {"color": "#c0392b", "ls": "-",  "lw": 2.2, "offset": 0.25,  "marker": "s", "ms": 6}
    }

    for label in ["Poisson", "NB", "PC", "NBM"]:
        if label in model_dict:
            st = styles[label]
            plt.plot(grid + st["offset"], model_dict[label], label=label, 
                     color=st["color"], linestyle=st["ls"], linewidth=st["lw"],
                     marker=st["marker"], markersize=st["ms"], alpha=0.9)

    plt.autoscale(enable=True, axis='y', tight=False)
    plt.title(f"PMF Fitting Analysis (Optimized View): {name}", fontsize=15, fontweight='bold', pad=20)
    plt.xlabel("Count (x)", fontsize=12); plt.ylabel("Probability", fontsize=12)
    plt.legend(loc='upper right', frameon=True, shadow=True, fontsize=10)
    plt.grid(axis='y', linestyle=':', alpha=0.4)
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()

def evaluate_one(name, data, extra_meta=None):
    data = np.asarray(data, dtype=int)
    xmax = xmax_from_data(data, q=0.999)
    grid = np.arange(xmax + 1)
    emp = empirical_pmf(data, xmax)
    rows, models_for_plot = [], {}

    # (1) Poisson
    mu, wP = poisson_baseline(data)
    p_pois = pmf_to_safe(wP(grid))
    rows.append({"model": "Poisson", "L1_diff": l1_sum_abs(emp, p_pois), "params": {"mu": mu}})
    models_for_plot["Poisson"] = p_pois

    # (2) PC (Poisson-Charlier)
    p_pc, info_pc = fit_pc_pmf(data, grid)
    p_pc = pmf_to_safe(p_pc)
    rows.append({"model": "PC", "L1_diff": l1_sum_abs(emp, p_pc), "params": info_pc})
    models_for_plot["PC"] = p_pc

    # (3) NB
    p_nb, info_nb = nb_pmf_mom_or_poisson(data, grid)
    p_nb = pmf_to_safe(p_nb)
    rows.append({"model": "NB", "L1_diff": l1_sum_abs(emp, p_nb), "params": info_nb})
    models_for_plot["NB"] = p_nb

    # (4) NBM (Negative Binomial Meixner)
    p_nbm, info_nbm = fit_meixner_pmf(data, grid)
    p_nbm = pmf_to_safe(p_nbm) if p_nbm is not None else p_nb.copy()
    rows.append({"model": "NBM", "L1_diff": l1_sum_abs(emp, p_nbm), "params": info_nbm if info_nbm else info_nb})
    models_for_plot["NBM"] = p_nbm

    # 순서 정렬: NBM -> NB -> PC -> Poisson
    order = ["NBM", "NB", "PC", "Poisson"]
    out = pd.DataFrame(rows)
    out['model'] = pd.Categorical(out['model'], categories=order, ordered=True)
    out = out.sort_values("model").reset_index(drop=True)

    summary = {"dataset": name, "n": len(data), "mean": np.mean(data), "var": np.var(data), "meta": extra_meta}
    return out, summary, grid, emp, models_for_plot

def write_report(report_path, all_tables, all_summaries):
    lines = []
    lines.append("=" * 115)
    lines.append(f"{'SNP DISCREPANCY & PARAMETER REPORT':^115}")
    lines.append("=" * 115)
    lines.append("-" * 115 + "\n")

    for summary, table in zip(all_summaries, all_tables):
        lines.append(f"▶ DATASET: {summary['dataset'].upper()}")
        lines.append(f"  Sample Stats: n={summary['n']:,} | Mean={summary['mean']:.4f} | Var={summary['var']:.4f}")
        lines.append("")
        lines.append("  [ 1. Discrepancy Table (L1 Diff) ]")
        lines.append("  " + "-" * 50)
        for _, row in table.iterrows():
            lines.append(f"  {row['model']:<25} | {row['L1_diff']:>18.6f}")
        lines.append("  " + "-" * 50 + "\n")
        lines.append("  [ 2. Parameter Details ]")
        lines.append("  " + "-" * 106)
        header = f"  {'Model':<25} | {'mu':>8} | {'beta':>8} | {'c':>8} | {'t1':>8} | {'t2':>8} | {'t3':>8} | {'t4':>8}"
        lines.append(header); lines.append("  " + "-" * 106)
        for _, row in table.iterrows():
            p = row['params']; mu = f"{float(p.get('mu', 0)):.4f}" if 'mu' in p else "-"
            beta = f"{float(p.get('beta', 0)):.4f}" if 'beta' in p else "-"
            c = f"{float(p.get('c', 0)):.4f}" if 'c' in p else "-"
            t_raw = p.get('theta', [0, 0, 0, 0, 0])
            t = [t_raw.get(str(i), 0) if isinstance(t_raw, dict) else t_raw[i] for i in range(5)]
            line = f"  {row['model']:<25} | {mu:>8} | {beta:>8} | {c:>8} | {t[1]:>8.4f} | {t[2]:>8.4f} | {t[3]:>8.4f} | {t[4]:>8.4f}"
            lines.append(line)
        lines.append("  " + "-" * 106 + "\n" + "." * 115 + "\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    ensure_dirs()
    fifa_csv, ins_csv = os.path.join(DATA_DIR, "results.csv"), os.path.join(DATA_DIR, "insurance.csv")
    fifa = load_fifa_counts(fifa_csv)
    ins, ins_meta = insurance_bimodal_to_count(ins_csv, bin_width=2000, truncate_p=92)

    all_tables, all_summaries = [], []
    for name, data, meta in [("FIFA", fifa, None), ("Insurance_Binned", ins, ins_meta)]:
        t, s, grid, emp, models = evaluate_one(name, data, extra_meta=meta)
        all_tables.append(t); all_summaries.append(s)
        plot_all(name, grid, emp, models, os.path.join(RESULT_DIR, f"plot_{name}.png"))

    write_report(os.path.join(RESULT_DIR, "report.txt"), all_tables, all_summaries)
    print(f"✅ 분석 완료! '{RESULT_DIR}' 폴더에서 리포트와 그래프를 확인하세요.")

if __name__ == "__main__":
    main()