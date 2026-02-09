import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# src 폴더 내부 모듈 임포트
from src.distributions import get_poisson_baseline, get_nb_baseline
from src.polynomials import get_charlier_polynomials, get_meixner_polynomials
from src.models import RigorousLinearTiltModel
from src.optimizer import RigorousOptimizer, get_rigorous_xmax

def run_fifa_analysis():
    # --- [경로 설정] ---
    DATA_DIR = r"D:\skewness_kurtosis\data"
    RESULT_DIR = r"D:\skewness_kurtosis\result_fifa" # FIFA 전용 결과 폴더
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # --- [분석 대상 정의] ---
    # 파일명, 라벨(식별용)
    targets = [
        ("results.csv", "FIFA-World-Cup")
    ]
    
    all_metrics = []

    for file_name, label in targets:
        path = os.path.join(DATA_DIR, file_name)
        if not os.path.exists(path):
            print(f"⚠️ 파일 없음: {path}")
            continue
            
        print(f"--- Analyzing {label} (Real World Cup Data) ---")
        
        # 데이터 로드
        df = pd.read_csv(path)

        # --- [FIFA 전용 전처리 로직] ---
        # 1. 월드컵 본선 경기만 추출
        df_wc = df[df['tournament'] == 'FIFA World Cup'].copy()
        # 2. 홈팀 + 원정팀 득점 합산 (Total Goals)
        data = (df_wc['home_score'] + df_wc['away_score']).dropna().values.astype(int)
        
        data = data[data >= 0]
        if len(data) == 0: 
            print(f"⚠️ {label}: 유효한 데이터가 없습니다.")
            continue

        n_samples = len(data)
        max_val = np.max(data)
        
        # 1. Poisson / PC (theta3, 4 최적화, theta2=0 고정)
        mu, p_pmf = get_poisson_baseline(data)
        ll_p_base = np.sum(np.log(np.maximum(p_pmf(data), 1e-12)))
        x_max_p = max(get_rigorous_xmax(mu, 'Poisson'), max_val)
        grid_p = np.arange(x_max_p + 1)
        psi_p = get_charlier_polynomials(grid_p, mu)
        model_pc = RigorousLinearTiltModel(p_pmf, psi_p)
        opt_pc = RigorousOptimizer(model_pc, data, [2, 3]) # theta3, theta4
        t_pc = opt_pc.optimize()
        ll_pc_total = ll_p_base + model_pc.get_log_likelihood(data, t_pc[[2,3]], [2,3])
        
        # 2. NB / NBM (theta3, 4 최적화, theta2=0 고정)
        nb_params, nb_pmf = get_nb_baseline(data)
        has_nb = nb_params is not None
        ll_nb_base, ll_nbm_total = None, None
        t_nbm = np.zeros(4)
        model_nbm = None
        
        if has_nb:
            ll_nb_base = np.sum(np.log(np.maximum(nb_pmf(data), 1e-12)))
            x_max_nb = max(get_rigorous_xmax(nb_params, 'NB'), max_val)
            grid_nb = np.arange(x_max_nb + 1)
            psi_nb = get_meixner_polynomials(grid_nb, *nb_params)
            model_nbm = RigorousLinearTiltModel(nb_pmf, psi_nb)
            opt_nbm = RigorousOptimizer(model_nbm, data, [2, 3])
            t_nbm = opt_nbm.optimize()
            ll_nbm_total = ll_nb_base + model_nbm.get_log_likelihood(data, t_nbm[[2,3]], [2,3])
            
        # 메트릭 취합
        m = {'Label': label, 'N': n_samples, 'P_LL': ll_p_base, 'PC_LL': ll_pc_total, 'PC_T3': t_pc[2], 'PC_T4': t_pc[3]}
        if has_nb:
            m.update({'NB_LL': ll_nb_base, 'NBM_LL': ll_nbm_total, 'NBM_T3': t_nbm[2], 'NBM_T4': t_nbm[3]})
        all_metrics.append(m)
        
        # --- [시각화] ---
        plt.figure(figsize=(8, 5)) 
        disp_max = int(np.percentile(data, 99.5)) 
        actual_disp = min(disp_max, max_val)
        x_disp = np.arange(actual_disp + 1)
        
        counts = np.bincount(data)
        y_obs = counts[:actual_disp + 1] / n_samples
        
        plt.bar(x_disp, y_obs, alpha=0.5, color='#7f8c8d', label='Empirical (FIFA WC)', edgecolor='none')
        plt.plot(x_disp, p_pmf(x_disp), 'g--', linewidth=1.2, label='Poisson', alpha=0.6)
        pmf_pc = model_pc.pmf()
        plt.plot(x_disp, pmf_pc[:actual_disp + 1], 'g-o', markersize=2, linewidth=0.8, label='PC Exp')
        
        if has_nb:
            plt.plot(x_disp, nb_pmf(x_disp), 'r--', linewidth=1.2, label='NB', alpha=0.6)
            pmf_nbm = model_nbm.pmf()
            plt.plot(x_disp, pmf_nbm[:actual_disp + 1], 'b-s', markersize=2, linewidth=0.8, label='NBM Exp')
            
        plt.title(f"Comparison: {label} (Real Goals)", fontsize=12, fontweight='bold')
        plt.xlabel("Total Goals per Match"); plt.ylabel("Probability")
        plt.legend(fontsize=9); plt.grid(alpha=0.15)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, f"plot_{label}.png"), dpi=300)
        plt.close()

    # --- [텍스트 리포트 생성] ---
    txt_path = os.path.join(RESULT_DIR, "fifa_comprehensive_report.txt")
    with open(txt_path, "w", encoding='utf-8') as f:
        divider = "="*145 + "\n"
        header = f"{'Dataset':<20} | {'Model':<12} | {'Total_LL':<13} | {'AIC':<13} | {'BIC':<13} | {'theta3':<9} | {'theta4':<9}\n"
        f.write(divider); f.write(header); f.write("-" * 145 + "\n")
        for m in all_metrics:
            ln_n = np.log(m['N'])
            # Poisson
            aic_p, bic_p = -2*m['P_LL']+2, -2*m['P_LL']+ln_n
            f.write(f"{m['Label']:<20} | {'Poisson':<12} | {m['P_LL']:13.2f} | {aic_p:13.2f} | {bic_p:13.2f} | {'-':<9} | {'-':<9}\n")
            # PC
            aic_pc, bic_pc = -2*m['PC_LL']+6, -2*m['PC_LL']+3*ln_n
            f.write(f"{'':<20} | {'PC':<12} | {m['PC_LL']:13.2f} | {aic_pc:13.2f} | {bic_pc:13.2f} | {m['PC_T3']:9.4f} | {m['PC_T4']:9.4f}\n")
            if 'NB_LL' in m:
                # NB
                aic_nb, bic_nb = -2*m['NB_LL']+4, -2*m['NB_LL']+2*ln_n
                f.write(f"{'':<20} | {'NB':<12} | {m['NB_LL']:13.2f} | {aic_nb:13.2f} | {bic_nb:13.2f} | {'-':<9} | {'-':<9}\n")
                # NBM
                aic_nbm, bic_nbm = -2*m['NBM_LL']+8, -2*m['NBM_LL']+4*ln_n
                f.write(f"{'':<20} | {'NBM':<12} | {m['NBM_LL']:13.2f} | {aic_nbm:13.2f} | {bic_nbm:13.2f} | {m['NBM_T3']:9.4f} | {m['NBM_T4']:9.4f}\n")
            f.write("-" * 145 + "\n")
        f.write(divider)
    print(f"✅ FIFA 분석 완료! {RESULT_DIR} 폴더를 확인하세요.")

if __name__ == "__main__":
    run_fifa_analysis()