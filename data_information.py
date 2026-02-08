import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import os

def generate_text_report():
    # --- [1. 경로 및 대상 설정] ---
    DATA_DIR = r"D:\skewness_kurtosis\data"
    RESULT_PATH = r"D:\skewness_kurtosis\result\data_characteristics_report.txt"
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)

    targets = [
        ("DoctorAUS.csv", "doctorco", "Doctor-Visits", None),
        ("insurance.csv", "charges", "Insurance-Bimodal", lambda x: np.round(x / 1000)),
        ("insurance_data.csv", "children", "Insurance-Children", None),
        ("LengthOfStay.csv", "lengthofstay", "Medical-LOS", None)
    ]

    report_lines = []
    
    # 헤더 정의
    header = f"{'Dataset':<20} | {'N':>8} | {'Mean':>7} | {'Var':>8} | {'V/M':>7} | {'Skew':>7} | {'Kurt':>7} | {'Zero%':>7} | {'Max':>5} | {'P99':>5}"
    divider = "-" * len(header)
    
    report_lines.append("=" * len(header))
    report_lines.append(f"{'Data Characteristics Final Report':^105}")
    report_lines.append("=" * len(header))
    report_lines.append(header)
    report_lines.append(divider)

    print("🚀 데이터 분석 중...")

    for file_name, col_name, label, transform_fn in targets:
        path = os.path.join(DATA_DIR, file_name)
        if not os.path.exists(path):
            print(f"⚠️ 파일 없음: {file_name}")
            continue

        df = pd.read_csv(path)
        
        # 데이터 추출 및 전처리
        if transform_fn:
            raw_data = transform_fn(df[col_name])
        else:
            raw_data = pd.to_numeric(df[col_name], errors='coerce')
            
        data = raw_data.dropna().values.astype(int)
        data = data[data >= 0]

        # 통계량 계산
        n = len(data)
        m = np.mean(data)
        v = np.var(data)
        disp = v / m if m > 0 else 0
        sk = skew(data)
        kt = kurtosis(data)
        z_prop = np.mean(data == 0) * 100
        mx = np.max(data)
        p99 = np.percentile(data, 99)

        # 텍스트 행 추가
        line = f"{label:<20} | {n:>8,d} | {m:>7.2f} | {v:>8.2f} | {disp:>7.2f} | {sk:>7.2f} | {kt:>7.2f} | {z_prop:>7.1f} | {mx:>5d} | {p99:>5.0f}"
        report_lines.append(line)

    report_lines.append("=" * len(header))
    
    # --- [2. 결과 출력 및 파일 저장] ---
    final_report = "\n".join(report_lines)
    
    # 콘솔 출력
    print("\n" + final_report)
    
    # 텍스트 파일 저장
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        f.write(final_report)
        
    print(f"\n✅ 텍스트 리포트 저장 완료: {RESULT_PATH}")

if __name__ == "__main__":
    generate_text_report()