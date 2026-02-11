import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks

def generate_final_count_data(n_samples=4000, seed=129):
    np.random.seed(seed)
    
    # 1. 혼합 비율 설정: 두 번째 봉우리가 더 높도록 4:6 비율
    n1 = int(n_samples * 0.4)
    n2 = n_samples - n1
    
    # Peak 1: 평균 20
    mu1, size1 = 20, 15
    group1 = stats.nbinom.rvs(n=size1, p=size1/(size1+mu1), size=n1)
    
    # Peak 2: 평균 140, 더 많은 샘플, 더 두꺼운 꼬리(size를 낮춤)
    mu2, size2 = 140, 5 
    group2 = stats.nbinom.rvs(n=size2, p=size2/(size2+mu2), size=n2)
    
    return np.concatenate([group1, group2])

# 데이터 생성
data = generate_final_count_data(4000)
df = pd.DataFrame({'count_value': data})

# --- 검증 및 리포트 ---
mean_val, var_val = np.mean(data), np.var(data)
kurt_val = stats.kurtosis(data)

# KDE 기반 봉우리 탐지
kde = stats.gaussian_kde(data)
x = np.linspace(0, data.max(), 4000)
peaks, _ = find_peaks(kde(x), prominence=kde(x).max() * 0.05)

print(f"1. 과분산 확인: {var_val/mean_val:.2f} (1보다 크면 과분산)")
print(f"2. 꼬리 두께 확인 (첨도): {kurt_val:.2f} (0보다 크면 두꺼운 꼬리)")
print(f"3. 봉우리 수: {len(peaks)}개 감지")

# 시각화
plt.figure(figsize=(12, 6))
sns.histplot(df['count_value'], kde=True, discrete=True, color='green')
plt.title('Final Bimodal Count Distribution (Peaks: 20 & 140)')
plt.show()