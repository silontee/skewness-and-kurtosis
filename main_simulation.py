import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import nbinom

# src 모듈 임포트
from src.distributions import get_poisson_baseline, get_nb_baseline
from src.polynomials import get_charlier_polynomials, get_meixner_polynomials
from src.models import RigorousLinearTiltModel
from src.optimizer import RigorousOptimizer, get_rigorous_xmax

def generate_dominant_peak_data(n=10000, p=0.92):
    """
    첫 번째 봉우리가 92%를 차지하는 'Dominant Peak' 시나리오
    Group 1: mu=10, var=15 (92% 비중) -> NB가 여기 완벽히 타겟팅됨
    Group 2: mu=50, var=100 (8% 비중) -> 아주 작은 두 번째 봉우리(혹) 형성
    """
    np.random.seed(403)
    # n=beta, p=1-c
    data_a = nbinom.rvs(n=20, p=0.666, size=int(n * p)) # mu=10
    data_b = nbinom.rvs(n=50, p=0.5, size=n - int(n * p)) # mu=50
    return np.concatenate([data_a, data_b])

def run_simulation_v5():
    RESULT_DIR = r"D:\skewness_kurtosis\result"
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # 1. 데이터 생성
    data = generate_dominant_peak_data()
    n_samples = len(data)
    max_val = np.max(data)
    
    # 2. Baseline & NBM Fitting
    nb_params, nb_pmf = get_nb_baseline(data)
    x_max_nb = max(get_rigorous_xmax(nb_params, 'NB'), max_val)
    grid_nb = np.arange(x_max_nb + 1)
    
    psi_nb = get_meixner_polynomials(grid_nb, *nb_params)
    model_nbm = RigorousLinearTiltModel(nb_pmf, psi_nb)
    opt_nbm = RigorousOptimizer(model_nbm, data, [2, 3]) # theta3, theta4 최적화
    t_nbm = opt_nbm.optimize()
    
    print(f"Simulation V5 - NBM Optimization Done.")

    # --- [시각화 최적화 - 원우님 스타일 반영] ---
    plt.figure(figsize=(8, 5)) 
    
    x_disp = np.arange(max_val + 1)
    counts = np.bincount(data)
    y_obs = counts / n_samples
    
    # 진한 회색 바 (#7f8c8d), 논문용 명암비
    plt.bar(x_disp, y_obs, alpha=0.6, color='#7f8c8d', label='Target (92% Dominant Peak)', edgecolor='none')
    
    # NB Baseline (빨간 점선) - 첫 번째 봉우리는 잘 맞지만, 뒤쪽 '혹'은 완전히 무시함
    plt.plot(x_disp, nb_pmf(x_disp), 'r--', linewidth=1.5, label='NB Baseline', alpha=0.7)
    
    # NBM (파란 실선) - 첫 봉우리를 유지하면서, 뒤쪽의 미세한 '혹'을 향해 선이 반응함
    pmf_nbm = model_nbm.pmf()
    plt.plot(x_disp, pmf_nbm[:max_val+1], 'b-s', markersize=2, linewidth=1.0, label='NBM (Bending to Tail)')
    
    plt.title("Simulation: NBM Capturing Minor Second Peak", fontsize=12, fontweight='bold')
    plt.xlabel("Count (x)"); plt.ylabel("Probability")
    plt.legend(fontsize=9, loc='upper right'); plt.grid(alpha=0.15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "plot_simulation_v5_dominant.png"), dpi=300)
    plt.show()

if __name__ == "__main__":
    run_simulation_v5()