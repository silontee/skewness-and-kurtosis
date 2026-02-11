# 왜도-첨도 직교 전개를 통한 카운트 분포 모델링

이산 확률질량함수(PMF)를 Poisson-Charlier 및 Meixner-음이항 직교다항식 전개로 특성화하는 반모수적(semi-nonparametric) 프레임워크.

**논문**: Lee, W.-W., Lee, J., Ha, H.-T. (2025). *Modeling Count Distributions via Skewness-Kurtosis Orthogonal Expansions*. Mathematics, MDPI.

---

## 연구 배경

### 문제

Poisson 분포 등 고전적 카운트 데이터 모형은 **등분산(equidispersion, 평균 = 분산)**을 가정합니다. 그러나 실제 데이터는 이 가정을 거의 만족하지 않습니다.

- **과분산(overdispersion)**: 분산이 평균보다 크다 (보험 청구, 교통사고 등)
- **영과잉(zero-inflation)**: 0이 비정상적으로 많다 (의료 방문 횟수 등)
- **두꺼운 꼬리(heavy tail)**: 극단값이 자주 발생한다 (바이러스 감염 건수 등)

### 해결 방법: Linear-Tilt 모델

기저 PMF `w(x)`에 직교다항식 보정을 곱하여, 데이터의 **왜도(skewness)**와 **첨도(kurtosis)**까지 맞추는 모델을 만듭니다.

```
p_θ(x) = w(x) × (1 + Σ θ_n ψ_n(x))    [논문 Eq (19)]
```

- `w(x)`: 기저 분포 (Poisson 또는 음이항)
- `ψ_n(x)`: 정규직교 다항식 (Charlier 또는 Meixner)
- `θ_n`: 데이터로부터 추정하는 전개 계수 (`θ_n = E_p[ψ_n(X)]`)

쉽게 말하면, 단순한 분포(Poisson/NB)로 대략적인 모양을 잡고, 직교다항식으로 세부 형태를 보정하는 방식입니다.

### 두 가지 전개 체계

| 체계 | 기저 분포 | 직교 다항식 | 매칭하는 적률 | 언제 사용? |
|------|----------|-----------|-------------|-----------|
| **Poisson-Charlier (PC)** | Poisson(μ) | Charlier | 평균만 (θ₁=0) | 분산 ≈ 평균 (V/M ≈ 1) |
| **Meixner-NB (NBM)** | NegBin(β, c) | Meixner | 평균+분산 (θ₁=θ₂=0) | 분산 > 평균 (V/M > 1) |

**V/M (Variance-to-Mean ratio)**이 핵심 판단 기준입니다:
- V/M ≈ 1 → PC 전개 사용
- V/M > 1 → NBM 전개 사용 (과분산이 클수록 NBM이 유리)

---

## 프로젝트 구조

```
skewness-and-kurtosis/
├── src/
│   ├── orthopoly.py        # Charlier/Meixner 정규직교 다항식 생성
│   ├── baselines.py        # Poisson/NB 기저 분포 및 파라미터 추정
│   ├── expansions_mom.py   # Linear-tilt PMF 피팅 (PC & NBM)
│   ├── metrics.py          # 경험적 PMF 생성, L1 거리 계산
│   ├── data_prep.py        # FIFA/보험 데이터 전처리
│   └── analysis.py         # 수렴도 분석 및 시각화 함수
├── main.py                 # 전체 분석 실행 (실제 데이터 + 시뮬레이션)
├── data/
│   ├── results.csv         # FIFA 월드컵 경기 결과
│   └── insurance.csv       # 의료 보험 청구금액
└── result/                 # 생성된 그래프 및 리포트
```

### 모듈별 역할

| 모듈 | 역할 | 주요 함수 |
|------|------|----------|
| `orthopoly.py` | 직교다항식 생성 (3항 재귀식 + 정규직교화) | `get_charlier_psi()`, `get_meixner_psi()` |
| `baselines.py` | 기저 분포 파라미터 추정 (적률법) | `poisson_baseline()`, `nb_moment_matched_params()` |
| `expansions_mom.py` | 전개 계수 추정 및 tilt PMF 생성 | `fit_pc_pmf()`, `fit_meixner_pmf()`, `normalize_pmf()` |
| `metrics.py` | 경험적 PMF 생성 및 적합도 측정 | `empirical_pmf()`, `l1_sum_abs()` |
| `data_prep.py` | 원시 데이터를 카운트 데이터로 변환 | `load_fifa_counts()`, `insurance_bimodal_to_count()` |
| `analysis.py` | 차수별 수렴 분석 및 비교 그래프 생성 | `run_convergence_study()`, `plot_comparison()`, `plot_pc_convergence()` |

### 데이터 처리 흐름

```
원시 데이터 (CSV)
    │
    ▼
[data_prep] 전처리 → 카운트 데이터 (정수 배열)
    │
    ├─→ [baselines] 기저 분포 파라미터 추정 (μ, β, c)
    │
    ├─→ [orthopoly] 직교다항식 ψ_n(x) 생성
    │
    ├─→ [expansions_mom] 전개 계수 θ_n 추정 → tilt PMF 생성
    │
    ├─→ [metrics] 경험적 PMF와 L1 거리 비교
    │
    └─→ [analysis] 시각화 및 리포트 생성
```

---

## 설치

```bash
git clone https://github.com/silontee/skewness-and-kurtosis.git
cd skewness-and-kurtosis

# uv (권장)
uv sync

# 또는 pip
pip install -e .
```

**요구사항**: Python >= 3.14, NumPy, SciPy, Pandas, Matplotlib, Seaborn

---

## 실행 방법

```bash
python main.py
```

`main.py` 하나로 아래 5가지 분석을 순서대로 실행합니다:

### 1부: 표준 4대 모델 비교 (Poisson vs NB vs PC vs NBM)

3개 데이터셋에 대해 4가지 모델을 적합하고 L1 거리로 비교합니다.

| 데이터셋 | 설명 | 특성 |
|---------|------|------|
| **FIFA** | 월드컵 경기당 총 골 수 | 약한 과분산 (V/M=1.32) |
| **Insurance** | 의료 보험 청구금액 (빈도화) | 강한 과분산 + 영과잉 (V/M=3.77) |
| **Simul_Heavy** | 이봉 NB 혼합 시뮬레이션 | 극단적 과분산 (V/M=23.5) |

### 2부: PC 전개 차수별 수렴 분석 (K = 0, 2, 4, 6, 8)

2개 데이터셋에 대해 PC 전개 차수를 높이면서 L1 오차 감소 추이를 관찰합니다.

| 데이터셋 | 설명 | 수렴 여부 |
|---------|------|----------|
| **Insurance** | 실제 보험 데이터 | 비단조 (과분산으로 인한 제약 위반) |
| **Simul_Success** | 이봉 Poisson 혼합 시뮬레이션 | 단조 감소 (이론 검증 성공) |

### 출력물

실행 후 `result/` 디렉토리에 다음 파일들이 생성됩니다:

| 파일 | 내용 |
|------|------|
| `plot_FIFA_std.png` | FIFA 4대 모델 비교 그래프 |
| `plot_Insurance_std.png` | 보험 4대 모델 비교 그래프 |
| `plot_Simul_Heavy_std.png` | 시뮬레이션(이봉 NB 혼합) 4대 모델 비교 그래프 |
| `plot_Insurance_convergence.png` | 보험 PC 차수별 수렴 그래프 |
| `plot_Simul_Success_convergence.png` | 시뮬레이션(이봉 Poisson 혼합) PC 수렴 그래프 |
| `unified_report.txt` | 전체 L1 오차 및 파라미터 상세 리포트 |

---

## 실험 결과

### FIFA 월드컵 골 수 (n=964, 평균=2.82, V/M=1.32)

| 모델 | L1 거리 | 비고 |
|------|---------|------|
| **NBM** | **0.0667** | 최우수 |
| NB | 0.0707 | |
| PC | 0.0724 | |
| Poisson | 0.1322 | |

V/M이 1에 가까워 PC도 잘 동작하지만, NB 기저로 분산까지 매칭한 NBM이 최고 성능을 보입니다.

### 보험 청구금액 (n=1338, 평균=3.81, V/M=3.77)

| 모델 | L1 거리 | 비고 |
|------|---------|------|
| **NB** | **0.2690** | 최우수 |
| NBM | 0.3005 | |
| PC | 0.5863 | |
| Poisson | 0.6786 | |

극단적 영과잉(zero-inflation)으로 고차 보정 시 **음수 확률**이 발생합니다. 이를 0으로 클리핑하는 과정에서 NBM의 적합도가 오히려 NB보다 떨어집니다. 논문의 feasible set C_K 제약 조건 위반과 관련된 현상입니다.

### 시뮬레이션: 이봉 NB 혼합 (n=4000, V/M=23.5)

| 모델 | L1 거리 | 비고 |
|------|---------|------|
| **NBM** | **0.4122** | 최우수 |
| NB | 0.5073 | |
| Poisson | 1.4779 | |
| PC | 1.5465 | |

극단적 과분산 데이터에서 NBM이 NB 대비 약 19% 개선. PC는 Poisson 기저의 V/M=1 가정이 완전히 무너져 성능이 최하위입니다.

### PC 수렴성 테스트 (이봉 Poisson 혼합, n=10000, V/M=2.13)

| 차수 | L1 거리 | 감소율 |
|------|---------|--------|
| K=0 (Poisson) | 0.4757 | - |
| K=2 | 0.1428 | -70.0% |
| K=4 | 0.0464 | -67.5% |
| K=6 | 0.0411 | -11.4% |
| K=8 | **0.0229** | -44.2% |

차수 증가에 따른 L1 오차의 **단조 감소**가 관찰됩니다. 이는 Parseval 항등식(논문 Proposition 1)의 실험적 검증입니다.

---

## 수학적 세부사항

### Charlier 다항식 (논문 Section 2.4)

하강 팩토리얼 전개 (Eq 23):

```
C_n(x; μ) = Σ_{k=0}^{n} C(n,k) (-1)^k (x)_k / μ^k
```

3항 재귀식 (코드에서 사용하는 계산 방식):

```
μ C_{n+1}(x) = (μ + n - x) C_n(x) - n C_{n-1}(x)
```

Poisson 가중 직교관계:

```
Σ_x w_P(x) C_n(x) C_m(x) = (n! / μ^n) δ_{nm}
```

정규직교 기저: `ψ_n(x) = C_n(x; μ) / √(n! / μ^n)`

### Meixner 다항식 (논문 Section 2.5)

초기하 함수 표현 (Eq 28):

```
M_n(x; β, c) = ₂F₁(-n, -x; β; 1 - 1/c)
```

3항 재귀식 (DLMF 표준):

```
c(β+n) M_{n+1}(x) = [(1+c)n + cβ - (1-c)x] M_n(x) - n M_{n-1}(x)
```

음이항 가중 직교관계:

```
Σ_x w_M(x) M_n(x) M_m(x) = (n! c^{-n} / (β)_n) δ_{nm}
```

정규직교 기저: `ψ_n(x) = M_n(x; β, c) / √h_n`,  `h_n = n! c^{-n} / (β)_n`

### NB 파라미터 추정 (Eq 29)

데이터의 평균 `μ_p`와 분산 `σ²_p`로부터 적률법(method of moments)으로 추정합니다.
과분산 조건(`σ²_p > μ_p`)이 필요합니다:

```
c = 1 - μ_p / σ²_p,    β = μ²_p / (σ²_p - μ_p)
```

### 오차 바운드 (Corollary 1)

근사 오차는 잔차 χ² 에너지로 상한이 결정됩니다:

```
TV(p, p*_K) ≤ ½ × (Σ_{n=K+1}^∞ (θ*_n)²)^{1/2}
```

K차까지 전개하면 잔차 에너지가 고차 계수들의 제곱합으로 수렴하므로, 차수를 높일수록 총변동 거리(TV)가 감소합니다.

---

## 구현 세부사항

### 계수 추정 방식

전개 계수는 정규직교 다항식의 **표본 기대값**으로 추정합니다 (Eq 17):

```python
# θ*_n = E_p[ψ_n(X)] 의 표본 추정
psi_at_data = get_charlier_psi(data, mu, K=K)
theta = np.mean(psi_at_data, axis=0)
```

이는 Proposition 2 (Eq 24-27)의 해석적 팩토리얼 적률 공식과 수학적으로 동치이며, 수치적으로 더 안정적입니다.

### 비음수성 처리

Linear-tilt 모델은 `θ ∉ C_K` (feasible set, Eq 20)일 때 `p_θ(x) < 0`을 생성할 수 있습니다.
현재 구현은 음수값을 0으로 클리핑하고 재정규화합니다:

```python
def normalize_pmf(p):
    p[p < 0] = 0.0
    return p / p.sum()
```

이 클리핑은 보험 데이터처럼 V/M이 큰 경우 오히려 적합도를 저하시킬 수 있습니다.

### 모델 선택 가이드

```
1. V/M 비율 계산: V/M = Var(X) / Mean(X)
   - V/M ≈ 1  → PC 전개 (Poisson 기저)
   - V/M > 1  → NBM 전개 (NB 기저)
   - V/M < 1  → Poisson 또는 Binomial 기저 (미구현)

2. 전개 차수 K 선택
   - K=4: 표준 (왜도 + 첨도 포착)
   - K=6, 8: 복잡한 다봉 분포에 사용
   - L1 오차가 증가하기 시작하면 과적합 → 차수 축소

3. 적합성 검증
   - 음수 확률 발생 비율 < 5% 권장
   - 높을 경우 차수 축소 또는 제약 최적화 고려
```

---

## 참고 문헌

1. Lee, W.-W., Lee, J., Ha, H.-T. (2025). Modeling Count Distributions via Skewness-Kurtosis Orthogonal Expansions. *Mathematics*, MDPI.
2. Ha, H.-T. (2024). Charlier Series Approximation for Nonhomogeneous Poisson Processes. *Commun. Stat. Appl. Methods*, 31, 645-659.
3. Im, J., Morikawa, K., Ha, H.-T. (2020). A Least Squares-Type Density Estimator Using a Polynomial Function. *Comput. Stat. Data Anal.*, 144, 106882.
4. Min, J., Provost, S.B., Ha, H.-T. (2009). Moment-Based Approximations of Probability Mass Functions. *Commun. Stat.-Theory Methods*, 38, 1969-1981.

---

## 라이선스

본 프로젝트는 위 학술 논문의 구현체입니다. 이론적 세부사항 및 증명은 논문을 참조하시기 바랍니다.
