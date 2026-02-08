import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import nbinom, poisson

# 기존 모듈 임포트
from src.distributions import get_poisson_baseline, get_nb_baseline
from src.polynomials import get_charlier_polynomials, get_meixner_polynomials
from src.models import RigorousLinearTiltModel
from src.optimizer import RigorousOptimizer, get_rigorous_xmax

# ==========================================
# 💡 [Patch Section] 수치 안정성 유지 (기존 로직 유지)
# ==========================================

class StableOptimizer(RigorousOptimizer):
    def optimize(self):
        x0 = np.full(len(self.active_indices), 1e-5)
        K_val = self.model.psi.shape[1] - 1
        bounds = [(-0.8, 0.8) for _ in range(len(self.active_indices))]

        def constraint(t_active):
            full_theta = np.zeros(K_val + 1)
            for i, idx in enumerate(self.active_indices):
                full_theta[idx] = t_active[i]
            z = 1.0 + np.dot(self.model.psi[:, 1:], full_theta[1:])
            return np.min(z) - 1e-9

        res = minimize(
            fun=lambda t: -self.model.get_log_likelihood(self.data, t, self.active_indices),
            x0=x0, method='SLSQP', bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint},
            options={'ftol': 1e-12, 'maxiter': 200}
        )
        final_theta = np.zeros(K_val + 1)
        for i, idx in enumerate(self.active_indices):
            final_theta[idx] = res.x[i]
        return final_theta

def patched_pmf(self):
    K_val = self.psi.shape[1] - 1
    t_vec = self.theta[1:] if len(self.theta) == K_val + 1 else self.theta
    tilt = 1.0 + np.dot(self.psi[:, 1:], t_vec)
    w_vals = self.w_func(np.arange(len(self.psi)))
    raw = np.maximum(w_vals * tilt, 0)
    return raw / np.sum(raw)

RigorousLinearTiltModel.pmf = patched_pmf

# ==========================================
# 💡 [Data Section] Ratio 2 데이터 생성 (기존 로직 유지)
# ==========================================

def generate_ratio2_data(n=10000):
    np.random.seed(403)
    data_mix = np.concatenate([
        poisson.rvs(mu=7, size=int(n * 0.5)),
        poisson.rvs(mu=13, size=n - int(n * 0.5))
    ])
    return data_mix

# ==========================================
# 💡 [Main Section] 파일명 및 리포트 수정
# ==========================================

def run_simulation():
    # 1. 경로 설정 (simulation_results)
    RESULT_DIR = r"D:\skewness_kurtosis\simulation_results"
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    print("🚀 시뮬레이션 실행 중...")
    data = generate_ratio2_data()
    n_samples, max_val = len(data), np.max(data)
    log_n = np.log(n_samples)
    
    # 2. Poisson & PC (theta2=0 Fixed)
    mu_base, p_pmf = get_poisson_baseline(data)
    grid_p = np.arange(max(get_rigorous_xmax(mu_base, 'Poisson'), max_val) + 1)
    psi_p = get_charlier_polynomials(grid_p, mu_base)
    model_pc = RigorousLinearTiltModel(p_pmf, psi_p)
    t_pc = StableOptimizer(model_pc, data, [2, 3]).optimize() # theta3, 4 최적화
    model_pc.theta = t_pc
    ll_p = np.sum(np.log(np.maximum(p_pmf(data), 1e-12)))
    ll_pc = ll_p + model_pc.get_log_likelihood(data, t_pc[[2, 3]], [2, 3])

    # 3. NB & NBM (theta2=0 Fixed)
    nb_params, nb_pmf = get_nb_baseline(data)
    grid_nb = np.arange(max(get_rigorous_xmax(nb_params, 'NB'), max_val) + 1)
    psi_nb = get_meixner_polynomials(grid_nb, *nb_params)
    model_nbm = RigorousLinearTiltModel(nb_pmf, psi_nb)
    t_nbm = StableOptimizer(model_nbm, data, [2, 3]).optimize() # theta3, 4 최적화
    model_nbm.theta = t_nbm
    ll_nb = np.sum(np.log(np.maximum(nb_pmf(data), 1e-12)))
    ll_nbm = ll_nb + model_nbm.get_log_likelihood(data, t_nbm[[2, 3]], [2, 3])

    # 4. 리포트 저장 (파일명: simulation_report.txt / theta 2,3,4 포함)
    report_path = os.path.join(RESULT_DIR, "simulation_report.txt")
    with open(report_path, "w", encoding='utf-8') as f:
        header = f"{'Model':<15} | {'LL':<12} | {'AIC':<10} | {'theta2':<8} | {'theta3':<8} | {'theta4':<8}"
        f.write("SNP Simulation Study Report (Ratio 2)\n" + "="*80 + "\n")
        f.write(header + "\n" + "-"*80 + "\n")
        
        # Base Lines
        f.write(f"{'Poisson':<15} | {ll_p:12.2f} | {-2*ll_p+2:10.2f} | {'-':<8} | {'-':<8} | {'-':<8}\n")
        f.write(f"{'NB':<15} | {ll_nb:12.2f} | {-2*ll_nb+4:10.2f} | {'-':<8} | {'-':<8} | {'-':<8}\n")
        
        # SNP Lines (theta2=0은 고정이므로 0.0000으로 표시)
        f.write(f"{'PC(SNP)':<15} | {ll_pc:12.2f} | {-2*ll_pc+8:10.2f} | {t_pc[1]:8.4f} | {t_pc[2]:8.4f} | {t_pc[3]:8.4f}\n")
        f.write(f"{'NBM(SNP)':<15} | {ll_nbm:12.2f} | {-2*ll_nbm+10:10.2f} | {t_nbm[1]:8.4f} | {t_nbm[2]:8.4f} | {t_nbm[3]:8.4f}\n")
        f.write("="*80 + "\n")

    # 5. 시각화 (파일명: simulation_plot.png / 기존 로직 그대로)
    plt.figure(figsize=(11, 7))
    x_plot = np.arange(max_val + 1)
    y_obs = np.bincount(data) / n_samples
    plt.bar(x_plot, y_obs, alpha=0.6, color='#7f8c8d', label='Ratio 2 Data', edgecolor='none')
    
    plt.plot(x_plot, p_pmf(x_plot), 'g--', linewidth=0.8, label='Poisson Baseline', alpha=0.6)
    plt.plot(x_plot, model_pc.pmf()[:max_val+1], 'g-o', linewidth=1.0, markersize=3, label='PC (SNP)')
    
    plt.plot(x_plot, nb_pmf(x_plot), 'r--', linewidth=0.8, label='NB Baseline', alpha=0.6)
    plt.plot(x_plot, model_nbm.pmf()[:max_val+1], 'b-s', linewidth=1.0, markersize=3, label='NBM (SNP)')
    
    plt.title("Simulation: Balanced Overdispersion (Variance $\\approx$ 2 * Mean)", fontsize=13, fontweight='bold')
    plt.xlabel("Count"); plt.ylabel("Probability"); plt.legend(frameon=False)
    plt.grid(axis='y', alpha=0.1); plt.tight_layout()
    
    # 그래프 저장
    plot_path = os.path.join(RESULT_DIR, "simulation_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.show()
    
    print(f"✅ 최종 파일 생성 완료: {RESULT_DIR}")

if __name__ == "__main__":
    run_simulation()