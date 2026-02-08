import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import nbinom, poisson

from src.distributions import get_poisson_baseline, get_nb_baseline
from src.polynomials import get_charlier_polynomials, get_meixner_polynomials
from src.models import RigorousLinearTiltModel
from src.optimizer import RigorousOptimizer, get_rigorous_xmax

# ==========================================
# 🛡️ [Stable Optimizer] 수치 안정성 극대화
# ==========================================

class StableOptimizer(RigorousOptimizer):
    def optimize(self):
        # 시작점: 아주 작은 값에서 출발
        x0 = np.full(len(self.active_indices), 1e-6)
        K_val = self.model.psi.shape[1] - 1
        # theta 값이 너무 커지면 꼬리가 찢어지므로 범위를 (-0.4, 0.4)로 제한
        bounds = [(-0.4, 0.4) for _ in range(len(self.active_indices))]

        def constraint(t_active):
            full_theta = np.zeros(K_val + 1)
            for i, idx in enumerate(self.active_indices):
                full_theta[idx] = t_active[i]
            # 확률 비음수 제약 (여유분 1e-8)
            z = 1.0 + np.dot(self.model.psi[:, 1:], full_theta[1:])
            return np.min(z) - 1e-8

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
    # 음수/0 방지용 최소값 설정
    raw = np.maximum(w_vals * tilt, 1e-18)
    return raw / np.sum(raw)

RigorousLinearTiltModel.pmf = patched_pmf

# ==========================================
# 💡 [Data Section] Real-style Count Data (Mixture 아님)
# ==========================================

def generate_stable_real_data(n=10000):
    """
    NB 기반에 미세한 Poisson Shifting을 주어 실제 데이터의 '불완전함' 구현
    Mixture 모델과는 철학적으로 다른 'Warped Distribution' 형태
    """
    np.random.seed(403)
    # 1. 베이스 NB (r=3, p=0.2 -> 평균 12)
    base_counts = nbinom.rvs(n=3, p=0.2, size=n)
    # 2. 미세한 Poisson 노이즈 (평균 0.5) 추가 - 데이터의 뾰족함을 뭉툭하게 만듦
    noise = poisson.rvs(mu=0.5, size=n)
    return base_counts + noise

# ==========================================
# 💡 [Main Section] 최종 실행
# ==========================================

def run_simulation():
    RESULT_DIR = r"D:\skewness_kurtosis\simulation_results"
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    print("🚀 실전형 데이터 기반 시뮬레이션 시작 (theta 2-4 FREE)...")
    data = generate_stable_real_data()
    n_samples, max_val = len(data), np.max(data)
    log_n = np.log(n_samples)

    # 1. Poisson & PC (FREE)
    mu_base, p_pmf = get_poisson_baseline(data)
    grid_p = np.arange(max(get_rigorous_xmax(mu_base, 'Poisson'), max_val) + 1)
    psi_p = get_charlier_polynomials(grid_p, mu_base)
    model_pc = RigorousLinearTiltModel(p_pmf, psi_p)
    t_pc = StableOptimizer(model_pc, data, [1, 2, 3]).optimize() # FREE
    model_pc.theta = t_pc
    ll_pc = np.sum(np.log(np.maximum(model_pc.pmf()[data], 1e-18)))

    # 2. NB & NBM (FREE)
    nb_params, nb_pmf = get_nb_baseline(data)
    grid_nb = np.arange(max(get_rigorous_xmax(nb_params, 'NB'), max_val) + 1)
    psi_nb = get_meixner_polynomials(grid_nb, *nb_params)
    model_nbm = RigorousLinearTiltModel(nb_pmf, psi_nb)
    t_nbm = StableOptimizer(model_nbm, data, [1, 2, 3]).optimize() # FREE
    model_nbm.theta = t_nbm
    ll_nbm = np.sum(np.log(np.maximum(model_nbm.pmf()[data], 1e-18)))

    # 3. 리포트 저장 (simulation_report.txt)
    report_path = os.path.join(RESULT_DIR, "simulation_report.txt")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write("SNP Simulation Final Report (Real-world Style Data)\n" + "="*85 + "\n")
        f.write(f"{'Model':<15} | {'LL':<12} | {'theta2':<8} | {'theta3':<8} | {'theta4':<8}\n")
        f.write("-" * 85 + "\n")
        f.write(f"{'PC(SNP)':<15} | {ll_pc:12.2f} | {t_pc[1]:8.4f} | {t_pc[2]:8.4f} | {t_pc[3]:8.4f}\n")
        f.write(f"{'NBM(SNP)':<15} | {ll_nbm:12.2f} | {t_nbm[1]:8.4f} | {t_nbm[2]:8.4f} | {t_nbm[3]:8.4f}\n")

    # 4. 그래프 저장 (simulation_plot.png)
    plt.figure(figsize=(11, 7))
    x_plot = np.arange(max_val + 1)
    plt.bar(x_plot, np.bincount(data)/n_samples, alpha=0.5, color='#34495e', label='Simulated Count Data')
    
    # 베이스라인 (점선)
    plt.plot(x_plot, p_pmf(x_plot), 'g--', alpha=0.4, label='Pois Baseline')
    plt.plot(x_plot, nb_pmf(x_plot), 'r--', alpha=0.4, label='NB Baseline')
    
    # SNP 확장 (실선)
    plt.plot(x_plot, model_pc.pmf()[:max_val+1], 'g-', linewidth=1.5, label='PC (SNP)')
    plt.plot(x_plot, model_nbm.pmf()[:max_val+1], 'b-', linewidth=1.5, label='NBM (SNP)')
    
    plt.yscale('log'); plt.legend(frameon=False); plt.grid(axis='y', alpha=0.1)
    plt.title("Simulation: Final Model Comparison (Theta 2-4 FREE)", fontsize=13, fontweight='bold')
    
    plot_path = os.path.join(RESULT_DIR, "simulation_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"✅ 최종 파일 생성 완료: {RESULT_DIR}")

if __name__ == "__main__":
    run_simulation()