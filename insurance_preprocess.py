import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 1. 데이터 불러오기
df = pd.read_csv('insurance.csv')
df_filtered = df[df['charges'] <= 30000].copy()

# 2. 구간 및 밀도 설계
low_limit = 14000
high_limit = 18000
edge_limit = 28000
max_val = df_filtered['charges'].max() + 1

# 구간별 막대(Bin) 개수 설정
low_bins = 20    
mid_bins = 15    
high_bins = 10
edge_bins = 5
total_bins_count = low_bins + mid_bins + high_bins + edge_bins + 1

# 경계선(Edges) 생성
edges = []
edges.extend(np.linspace(0, low_limit, low_bins + 1))
edges.extend(np.linspace(low_limit, high_limit, mid_bins + 1)[1:])
edges.extend(np.linspace(high_limit, edge_limit, high_bins + 1)[1:])
edges.extend(np.linspace(edge_limit, max_val, edge_bins + 1)[1:])
edges = sorted(list(set(edges)))

# 3. pd.cut을 활용한 데이터 전처리
df_filtered['balanced_bin_index'] = pd.cut(df_filtered['charges'], bins=edges, labels=False, include_lowest=True)

# 4. 시각화
plt.figure(figsize=(12, 6))

counts = df_filtered['balanced_bin_index'].value_counts().sort_index()
data = df_filtered['balanced_bin_index']
kde = gaussian_kde(data)
x_range = np.linspace(data.min(), data.max(), 1000)
y_density = kde(x_range)
bin_edges = np.arange(total_bins_count + 1) - 0.5

plt.plot(x_range, y_density, color='red', linewidth=2, label='KDE Curve')
plt.hist(data, density=True, bins=bin_edges, color='green', edgecolor='white', alpha=0.4, label='Histogram')
plt.title('Balanced Bimodal Distribution')
plt.xlabel('Custom Bin Index')
plt.ylabel('Frequency')
plt.show()

# 5. 가공된 데이터 저장
df_filtered.to_csv('insurance_bimodal.csv', index=False)