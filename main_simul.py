# main_simul.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from src.expansions_mom import fit_meixner_pmf, fit_pc_pmf
from src.metrics import l1_sum_abs, empirical_pmf

def generate_final_count_data(n_samples=4000, seed=129):
    np.random.seed(seed)
    n1 = int(n_samples * 0.4)
    n2 = n_samples - n1
    g1 = stats.nbinom.rvs(n=15, p=15/(15+20), size=n1)
    g2 = stats.nbinom.rvs(n=5, p=5/(5+140), size=n2)
    return np.concatenate([g1, g2])

def run_simulation():
    data = generate_final_count_data()
    mu, var = np.mean(data), np.var(data)
    xmax = int(np.max(data) + 10)
    grid = np.arange(xmax + 1)
    emp = empirical_pmf(data, xmax)

    # NBM vs PC-Order 4 비교
    p_pc, _ = fit_pc_pmf(data, grid)
    p_nbm, _ = fit_meixner_pmf(data, grid)

    plt.figure(figsize=(12, 6))
    sns.histplot(data, bins=xmax, stat="probability", alpha=0.3, color='green', label='Simulation Data')
    plt.plot(grid, p_pc, label=f"PC Order 4 (L1: {l1_sum_abs(emp, p_pc):.4f})", color='blue')
    plt.plot(grid, p_nbm, label=f"NBM (L1: {l1_sum_abs(emp, p_nbm):.4f})", color='red', lw=2)
    plt.title(f'Bimodal Simulation (V/M: {var/mu:.2f}) - NBM Superiority')
    plt.legend(); plt.savefig("result/plot_simulation_nbm_win.png"); plt.show()
    
    print(f"Simulation Analysis Done. NBM L1: {l1_sum_abs(emp, p_nbm):.4f}")

if __name__ == "__main__": run_simulation()