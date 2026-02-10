# src/data_prep.py
import numpy as np
import pandas as pd

def load_fifa_counts(results_csv_path: str):
    df = pd.read_csv(results_csv_path)
    df_wc = df[df["tournament"] == "FIFA World Cup"].copy()
    x = (df_wc["home_score"] + df_wc["away_score"]).dropna().astype(int).to_numpy()
    return x[x >= 0]

def insurance_bimodal_to_count(insurance_csv_path: str, col="charges", bin_width=1200):
    """
    보험 데이터를 고정된 너비(bin_width)로 나누어 카운트 데이터로 변환.
    - bin_width: 1200 (NBM이 이봉성을 인지할 수 있는 최적의 해상도)
    - 상위 5% 아웃라이어 제거로 4차 적률(theta4) 안정화
    """
    df = pd.read_csv(insurance_csv_path)
    raw_x = pd.to_numeric(df[col], errors="coerce").dropna().astype(float).to_numpy()

    # 1. 아웃라이어 제거 (95분위수 컷)
    upper_limit = np.percentile(raw_x, 95)
    filtered_x = raw_x[raw_x <= upper_limit]

    # 2. 고정 폭으로 나누기 (// 연산 사용)
    cats = (filtered_x // bin_width).astype(int)
    
    meta = {
        "bin_width": bin_width, 
        "upper_limit": round(upper_limit, 2), 
        "max_x": int(cats.max())
    }
    return cats, meta