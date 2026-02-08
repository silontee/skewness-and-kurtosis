import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 원본 모듈 임포트
from src.distributions import get_poisson_baseline, get_nb_baseline
from src.polynomials import get_charlier_polynomials, get_meixner_polynomials
from src.models import RigorousLinearTiltModel
from src.optimizer import RigorousOptimizer, get_rigorous_xmax

# --- [Patch Section: 이전과 동일] ---
class ExperimentalOptimizer(RigorousOptimizer):
    def optimize(self):
        x0 = np.zeros(len(self.active_indices))
        K_val = self.model.psi.shape[1] - 1
        def constraint(t_active):
            full_theta = np.zeros(K_val + 1)
            for i, idx in enumerate(self.active_indices):
                full_theta[idx] = t_active[i]
            return 1.0 + np.dot(self.model.psi[:, 1:], full_theta[1:])
        res = minimize(
            fun=lambda t: -self.model.get_log_likelihood(self.data, t, self.active_indices),
            x0=x0, method='SLSQP',
            constraints={'type': 'ineq', 'fun': constraint},
            options={'ftol': 1e-11}
        )
        final_theta = np.zeros(K_val + 1)
        for i, idx in enumerate(self.active_indices):
            final_theta[idx] = res.x[i]
        return final_theta

def patched_get_log_likelihood(self, data, active_theta, active_indices):
    K_val = self.psi.shape[1] - 1
    full_theta = np.zeros(K_val + 1)
    for i, idx in enumerate(active_indices):
        full_theta[idx] = active_theta[i]
    z_vals = 1.0 + np.dot(self.psi[data, 1:], full_theta[1:])
    if np.any(z_vals <= 0): return -1e15
    return np.sum(np.log(z_vals))

def patched_pmf(self):
    K_expected = self.psi.shape[1] - 1
    t_vec = self.theta[1:] if len(self.theta) == K_expected + 1 else self.theta[:K_expected]
    tilt = 1.0 + np.dot(self.psi[:, 1:], t_vec)
    w_vals = self.w_func(np.arange(len(self.psi)))
    return w_vals * tilt

RigorousLinearTiltModel.get_log_likelihood = patched_get_log_likelihood
RigorousLinearTiltModel.pmf = patched_pmf

# --- [Main Section] ---
def run_final_selected_analysis():
    DATA_DIR = r"D:\skewness_kurtosis\data"
    RESULT_DIR = r"D:\skewness_kurtosis\result_theta2_non0"
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    targets = [
        ("insurance_data.csv", "children", "Insurance-Children"),
        ("DoctorAUS.csv", "doctorco", "Doctor-Visits"),
        ("LengthOfStay.csv", "lengthofstay", "Medical-LOS"),
        ("insurance.csv", "charges", "Insurance-Bimodal")
    ]
    
    all_metrics = []

    for file_name, col_name, label in targets:
        path = os.path.join(DATA_DIR, file_name)
        if not os.path.exists(path): continue
        print(f"\n>>> 분석 중: {label}...")
        df = pd.read_csv(path)

        if label == "Insurance-Bimodal":
            data = np.round(df['charges'] / 1000).astype(int)
        else:
            data = pd.to_numeric(df[col_name], errors='coerce').dropna().values.astype(int)
        
        data = data[data >= 0]
        n, log_n = len(data), np.log(len(data))
        max_val = np.max(data)
        
        # 1. Poisson / PC 최적화
        mu, p_pmf = get_poisson_baseline(data)
        x_max_p = max(get_rigorous_xmax(mu, 'Poisson'), max_val)
        grid_p = np.arange(x_max_p + 1)
        psi_p = get_charlier_polynomials(grid_p, mu)
        model_pc = RigorousLinearTiltModel(p_pmf, psi_p)
        t_pc = ExperimentalOptimizer(model_pc, data, [2, 3, 4]).optimize()
        ll_p_base = np.sum(np.log(np.maximum(p_pmf(data), 1e-12)))
        ll_pc = ll_p_base + model_pc.get_log_likelihood(data, t_pc[[2, 3, 4]], [2, 3, 4])

        # 2. NB / NBM 최적화
        nb_params, nb_pmf = get_nb_baseline(data)
        has_nb = nb_params is not None
        ll_nb_base, ll_nbm, t_nbm = None, None, None
        if has_nb:
            ll_nb_base = np.sum(np.log(np.maximum(nb_pmf(data), 1e-12)))
            x_max_nb = max(get_rigorous_xmax(nb_params, 'NB'), max_val)
            grid_nb = np.arange(x_max_nb + 1)
            psi_nb = get_meixner_polynomials(grid_nb, *nb_params)
            model_nbm = RigorousLinearTiltModel(nb_pmf, psi_nb)
            t_nbm = ExperimentalOptimizer(model_nbm, data, [2, 3, 4]).optimize()
            ll_nbm = ll_nb_base + model_nbm.get_log_likelihood(data, t_nbm[[2, 3, 4]], [2, 3, 4])

        all_metrics.append({'Label': label, 'N': n, 'P_LL': ll_p_base, 'PC_LL': ll_pc, 'PC_T2': t_pc[2], 'PC_T3': t_pc[3], 'PC_T4': t_pc[4],
                           'NB_LL': ll_nb_base, 'NBM_LL': ll_nbm, 'NBM_T2': t_nbm[2], 'NBM_T3': t_nbm[3], 'NBM_T4': t_nbm[4]})

        # --- [그래프 스타일 수정 섹션] ---
        plt.figure(figsize=(10, 6))
        disp_max = int(np.percentile(data, 99.5))
        x_plot = np.arange(min(disp_max, max_val) + 1)
        plt.bar(x_plot, np.bincount(data)[:len(x_plot)]/n, alpha=0.3, color='#95a5a6', label='Empirical')
        
        # 💡 선은 얇게(linewidth=0.8), 각 점마다 마커(marker) 추가
        plt.plot(x_plot, p_pmf(x_plot), 'g--', linewidth=0.6, marker='o', markersize=3, label='Poisson Baseline', alpha=0.5)
        
        model_pc.theta = t_pc
        plt.plot(x_plot, model_pc.pmf()[:len(x_plot)], 'g-', linewidth=0.8, marker='s', markersize=3, label='PC (Relaxed)')
        
        if has_nb:
            plt.plot(x_plot, nb_pmf(x_plot), 'r--', linewidth=0.6, marker='o', markersize=3, label='NB Baseline', alpha=0.5)
            model_nbm.theta = t_nbm
            plt.plot(x_plot, model_nbm.pmf()[:len(x_plot)], 'b-', linewidth=0.8, marker='^', markersize=3, label='NBM (Relaxed)')
            
        plt.title(f"Comparison: {label} (Discrete Plot Style)", fontsize=13, fontweight='bold')
        plt.xlabel("Count (x)"); plt.ylabel("Probability"); plt.legend(); plt.grid(alpha=0.15)
        plt.savefig(os.path.join(RESULT_DIR, f"plot_{label}.png"), dpi=300); plt.close()

    # --- [텍스트 리포트 생성 및 저장] ---
    report_path = os.path.join(RESULT_DIR, "final_comprehensive_report.txt")
    with open(report_path, "w", encoding='utf-8') as f:
        header = f"{'Dataset':<20} | {'Model':<10} | {'Total_LL':<12} | {'AIC':<10} | {'BIC':<10} | {'t2':<7} | {'t3':<7} | {'t4':<7}"
        f.write(header + "\n" + "="*125 + "\n")
        for m in all_metrics:
            ln_n = np.log(m['N'])
            f.write(f"{m['Label']:<20} | {'Poisson':<10} | {m['P_LL']:12.2f} | {-2*m['P_LL']+2:10.2f} | {-2*m['P_LL']+ln_n:10.2f} | {'-':<7} | {'-':<7} | {'-':<7}\n")
            f.write(f"{'':<20} | {'PC(R)':<10} | {m['PC_LL']:12.2f} | {-2*m['PC_LL']+8:10.2f} | {-2*m['PC_LL']+4*ln_n:10.2f} | {m['PC_T2']:7.3f} | {m['PC_T3']:7.3f} | {m['PC_T4']:7.3f}\n")
            if m['NB_LL']:
                f.write(f"{'':<20} | {'NB':<10} | {m['NB_LL']:12.2f} | {-2*m['NB_LL']+4:10.2f} | {-2*m['NB_LL']+2*ln_n:10.2f} | {'-':<7} | {'-':<7} | {'-':<7}\n")
                f.write(f"{'':<20} | {'NBM(R)':<10} | {m['NBM_LL']:12.2f} | {-2*m['NBM_LL']+10:10.2f} | {-2*m['NBM_LL']+5*ln_n:10.2f} | {m['NBM_T2']:7.3f} | {m['NBM_T3']:7.3f} | {m['NBM_T4']:7.3f}\n")
            f.write("-" * 125 + "\n")
    print(f"✅ 분석 완료! 결과는 {RESULT_DIR} 폴더에 저장되었습니다.")

if __name__ == "__main__":
    run_final_selected_analysis()