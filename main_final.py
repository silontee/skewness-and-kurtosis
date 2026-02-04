import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# src 폴더 내부 모듈 임포트
from src.distributions import get_poisson_baseline, get_nb_baseline
from src.polynomials import get_charlier_polynomials, get_meixner_polynomials
from src.models import RigorousLinearTiltModel
from src.optimizer import RigorousOptimizer, get_rigorous_xmax

def run_final_analysis():
    # --- [경로 설정] ---
    DATA_DIR = r"D:\skewness_kurtosis\data"
    RESULT_DIR = r"D:\skewness_kurtosis\result"
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # 분석 대상 데이터셋 정의 (NBA는 총 득점과 총 어시스트로 분화)
    targets = [
        ("LengthOfStay.csv", "lengthofstay", "Medical-LOS"),
        ("olist_order_payments_dataset.csv", "payment_installments", "Olist-Installments"),
        ("all_seasons.csv", "total_pts", "NBA-Total-Pts"),
        ("all_seasons.csv", "total_ast", "NBA-Total-Ast")
    ]
    
    all_metrics = []

    for file_name, col_name, label in targets:
        path = os.path.join(DATA_DIR, file_name)
        if not os.path.exists(path):
            print(f"⚠️ 파일 없음: {path}")
            continue
            
        print(f"--- Analyzing {label} ---")
        df = pd.read_csv(path)
        
        # 💡 NBA 데이터 전처리: 평균 수치를 총합 카운트로 변환
        if label.startswith("NBA"):
            if "total_pts" in col_name:
                data = np.round(df['gp'] * df['pts']).astype(int)
            else:
                data = np.round(df['gp'] * df['ast']).astype(int)
        else:
            data = df[col_name].dropna().values.astype(int)
        
        data = data[data >= 0]
        n_samples = len(data)
        max_val = np.max(data)
        
        # 1. Poisson / PC (Isolation: theta3, 4)
        mu, p_pmf = get_poisson_baseline(data)
        ll_p_base = np.sum(np.log(np.maximum(p_pmf(data), 1e-12)))
        x_max_p = max(get_rigorous_xmax(mu, 'Poisson'), max_val)
        grid_p = np.arange(x_max_p + 1)
        psi_p = get_charlier_polynomials(grid_p, mu)
        model_pc = RigorousLinearTiltModel(p_pmf, psi_p)
        opt_pc = RigorousOptimizer(model_pc, data, [2, 3])
        t_pc = opt_pc.optimize()
        ll_pc_total = ll_p_base + model_pc.get_log_likelihood(data, t_pc[[2,3]], [2,3])
        
        # 2. NB / NBM (Isolation: theta3, 4)
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
            
        m = {'Label': label, 'N': n_samples, 'P_LL': ll_p_base, 'PC_LL': ll_pc_total, 'PC_T3': t_pc[2], 'PC_T4': t_pc[3]}
        if has_nb:
            m.update({'NB_LL': ll_nb_base, 'NBM_LL': ll_nbm_total, 'NBM_T3': t_nbm[2], 'NBM_T4': t_nbm[3]})
        all_metrics.append(m)
        
        # --- 시각화 ---
        plt.figure(figsize=(10, 6))
        # NBA 득점 데이터 등 범위가 큰 데이터를 위해 퍼센타일 기반 범위 설정
        disp_max = int(np.percentile(data, 99)) 
        actual_disp = min(disp_max, max_val)
        x_disp = np.arange(actual_disp + 1)
        
        counts = np.bincount(data)
        y_obs = counts[:actual_disp + 1] / n_samples
        
        plt.bar(x_disp, y_obs, alpha=0.3, color='#bdc3c7', label='Empirical', edgecolor='white')
        plt.plot(x_disp, p_pmf(x_disp), 'g--', label='Poisson', alpha=0.7)
        pmf_pc = model_pc.pmf()
        plt.plot(x_disp, pmf_pc[:actual_disp + 1], 'g-o', markersize=3, label='PC Exp')
        
        if has_nb:
            plt.plot(x_disp, nb_pmf(x_disp), 'r--', label='NB', alpha=0.7)
            pmf_nbm = model_nbm.pmf()
            plt.plot(x_disp, pmf_nbm[:actual_disp + 1], 'b-s', markersize=4, label='NBM Exp')
            
        plt.title(f"Comparison: {label}", fontsize=14, fontweight='bold')
        plt.xlabel("Count (x)"); plt.ylabel("Probability")
        plt.legend(); plt.grid(alpha=0.2)
        plt.savefig(os.path.join(RESULT_DIR, f"plot_{label}.png"), dpi=300)
        plt.close()

    # --- 텍스트 리포트 저장 (BIC 추가 버전) ---
    txt_path = os.path.join(RESULT_DIR, "analysis_report_v2.txt")
    with open(txt_path, "w", encoding='utf-8') as f:
        divider = "="*145 + "\n"
        header = f"{'Dataset':<20} | {'Model':<12} | {'Total_LL':<13} | {'AIC':<13} | {'BIC':<13} | {'theta3':<9} | {'theta4':<9}\n"
        f.write(divider); f.write(header); f.write("-" * 145 + "\n")
        
        for m in all_metrics:
            d = m['Label']
            n = m['N']
            ln_n = np.log(n)
            
            # Poisson (k=1)
            aic_p = -2*m['P_LL'] + 2*1
            bic_p = -2*m['P_LL'] + 1*ln_n
            f.write(f"{d:<20} | {'Poisson':<12} | {m['P_LL']:13.2f} | {aic_p:13.2f} | {bic_p:13.2f} | {'-':<9} | {'-':<9}\n")
            
            # PC (k=3)
            aic_pc = -2*m['PC_LL'] + 2*3
            bic_pc = -2*m['PC_LL'] + 3*ln_n
            f.write(f"{'':<20} | {'PC':<12} | {m['PC_LL']:13.2f} | {aic_pc:13.2f} | {bic_pc:13.2f} | {m['PC_T3']:9.4f} | {m['PC_T4']:9.4f}\n")
            
            if 'NB_LL' in m:
                # NB (k=2)
                aic_nb = -2*m['NB_LL'] + 2*2
                bic_nb = -2*m['NB_LL'] + 2*ln_n
                f.write(f"{'':<20} | {'NB':<12} | {m['NB_LL']:13.2f} | {aic_nb:13.2f} | {bic_nb:13.2f} | {'-':<9} | {'-':<9}\n")
                
                # NBM (k=4)
                aic_nbm = -2*m['NBM_LL'] + 2*4
                bic_nbm = -2*m['NBM_LL'] + 4*ln_n
                f.write(f"{'':<20} | {'NBM':<12} | {m['NBM_LL']:13.2f} | {aic_nbm:13.2f} | {bic_nbm:13.2f} | {m['NBM_T3']:9.4f} | {m['NBM_T4']:9.4f}\n")
            f.write("-" * 145 + "\n")
        f.write(divider)

if __name__ == "__main__":
    run_final_analysis()