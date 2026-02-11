# src/data_prep.py
import numpy as np
import pandas as pd

def load_fifa_counts(results_csv_path: str):
    df = pd.read_csv(results_csv_path)
    df_wc = df[df["tournament"] == "FIFA World Cup"].copy()
    x = (df_wc["home_score"] + df_wc["away_score"]).dropna().astype(int).to_numpy()
    return x[x >= 0]

def insurance_bimodal_to_count(insurance_csv_path: str, col="charges", bin_width=2000, truncate_p=92):
    """
    SNP 모델의 수렴 안정성을 위해 상위 아웃라이어를 절단(Truncation)하는 방식.
    - bin_width: 2000 (과분산 수치를 2.5 내외로 조절하여 PC 모델 안정화)
    - truncate_p: 92 (상위 8% 제거로 4차 적률의 폭주 방지)
    """
    df = pd.read_csv(insurance_csv_path)
    raw_x = pd.to_numeric(df[col], errors="coerce").dropna().astype(float).to_numpy()

    # 1. 아웃라이어 절단 (92% 지점에서 컷)
    upper_limit = np.percentile(raw_x, truncate_p)
    filtered_x = raw_x[raw_x <= upper_limit]

    # 2. 고정 폭으로 나누어 카운트 생성
    cats = (filtered_x // bin_width).astype(int)
    
    meta = {
        "method": f"Truncation at {truncate_p}%",
        "bin_width": bin_width,
        "max_x": int(cats.max())
    }
    return cats, meta