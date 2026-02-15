import numpy as np
import pandas as pd

# 1. FIFA 데이터 로드
def load_fifa_counts(results_csv_path: str):
    df = pd.read_csv(results_csv_path)
    df_wc = df[df["tournament"] == "FIFA World Cup"].copy()
    x = (df_wc["home_score"] + df_wc["away_score"]).dropna().astype(int).to_numpy()
    return x[x >= 0]


# 2. Insurance 데이터 로드
def insurance_bimodal_to_count(insurance_csv_path: str, col="charges", bin_width=1500):
    """
    연속형 데이터를 bin_width: 1000 으로 count data로 변환
    """
    df = pd.read_csv(insurance_csv_path)
    x = pd.to_numeric(df[col], errors="coerce").dropna().astype(float).to_numpy()

    # Binning
    cats = (x // bin_width).astype(int)
    return cats

# 3. Sepsis HR Count 데이터 로드
def load_sepsis_hr_counts(csv_path: str, cap_p=99.5):
    """
    Sepsis 데이터의 환자별 HR 측정 횟수 추출
    - 상위 0.5% 데이터 제거 (cap_p=99.5)
    """
    df = pd.read_csv(csv_path)
    # Patient_ID별로 실제 HR(심박수) 값이 존재하는 행의 개수 산출
    hr_counts = df.groupby('Patient_ID')['HR'].count().values
    
    # 97th Percentile Capping
    hr_cap = np.percentile(hr_counts, cap_p)
    data_hr = hr_counts[hr_counts <= hr_cap]
    return data_hr

# 4. Sepsis Lab Count 데이터 로드
def load_sepsis_lab_counts(csv_path: str, cap_p=99.5):
    """
    Sepsis 데이터의 환자별 Lab 검사 총합 횟수 추출
    - 상위 0.5% 데이터 제거 (cap_p=99.5)
    """
    lab_cols = [
        'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
        'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
        'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
        'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
        'Fibrinogen', 'Platelets'
    ]
    df = pd.read_csv(csv_path)
    # 환자별로 모든 Lab 컬럼에서 값이 있는(Non-null) 횟수를 합산
    lab_counts = df.groupby('Patient_ID')[lab_cols].count().sum(axis=1).values
    
    # 99th Percentile Capping
    lab_cap = np.percentile(lab_counts, cap_p)
    data_lab = lab_counts[lab_counts <= lab_cap]
    return data_lab

# 5. Tumor Size 데이터 로드
def load_tumor_size_counts(csv_path: str, bin_width=5):
    """
    SEER 데이터의 종양 크기 전처리
    - 5mm 단위 Binning 적용
    """
    df = pd.read_csv(csv_path)
    # Tumor Size 컬럼 추출 및 결측치 제거
    x = df['Tumor Size'].dropna().to_numpy()
    
    # 5mm 단위로 나누어 정수화
    x = (x // bin_width).astype(int)
    
    return x[x >= 0]