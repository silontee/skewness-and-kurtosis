import numpy as np
import pandas as pd
# FIFA데이터만 선택
def load_fifa_counts(results_csv_path: str):
    df = pd.read_csv(results_csv_path)
    df_wc = df[df["tournament"] == "FIFA World Cup"].copy()
    x = (df_wc["home_score"] + df_wc["away_score"]).dropna().astype(int).to_numpy()
    return x[x >= 0]


def insurance_bimodal_to_count(insurance_csv_path: str, col="charges", bin_width=3000, cap_p=95):
    """
    꼬리 데이터를 삭제하지 않고 캡핑(Capping)하여 포함하는 전처리.
    - bin_width: 3000 (V/M Ratio를 PC 모델 안정권인 2.5 내외로 조절)
    - cap_p: 95 (상위 5% 데이터를 버리지 않고 임계값으로 고정)
    """
    df = pd.read_csv(insurance_csv_path)
    raw_x = pd.to_numeric(df[col], errors="coerce").dropna().astype(float).to_numpy()

    # 1. Winsorization (Capping): 꼬리 데이터를 삭제하지 않고 포함
    cap_value = np.percentile(raw_x, cap_p)
    capped_x = np.where(raw_x > cap_value, cap_value, raw_x)

    # 2. 고정 폭 Binning
    cats = (capped_x // bin_width).astype(int)
    
    mean_val = np.mean(cats)
    var_val = np.var(cats)
    
    meta = {
        "method": f"Winsorization at {cap_p}%",
        "bin_width": bin_width,
        "vm_ratio": round(var_val / (mean_val + 1e-10), 4),
        "max_x": int(cats.max())
    }
    return cats, meta