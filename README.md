# 왜도-첨도 직교 전개를 통한 카운트 분포 모델링

이산 확률질량함수(PMF)를 Poisson-Charlier 및 Meixner-음이항 직교다항식 전개로 특성화하는 반모수적(semi-nonparametric) 프레임워크.

**논문**: Lee, W.-W., Lee, J., Ha, H.-T. (2025). *Modeling Count Distributions via Skewness-Kurtosis Orthogonal Expansions*. Mathematics, MDPI.

---

## 연구 배경

Poisson 분포 등 고전적 카운트 데이터 모형은 **등분산(equidispersion, 평균 = 분산)**을 가정하지만, 보험 청구, 입원 일수, 바이러스 감염 건수 등 실제 데이터는 **과분산(overdispersion)**, 영과잉(zero-inflation), 두꺼운 꼬리(heavy tail) 등 복잡한 구조를 보입니다.

본 프로젝트는 기저 분포를 직교다항식으로 보정하여 **왜도(skewness)**와 **첨도(kurtosis)**까지 매칭하는 **선형 기울기(linear-tilt) 모델**을 구현합니다.

### 핵심 아이디어

기저 PMF `w(x)` (Poisson 또는 음이항)를 직교다항식으로 보정합니다:

```
p_θ(x) = w(x) × (1 + Σ θ_n ψ_n(x))    [논문 Eq (19)]
```

- `ψ_n(x)`: 정규직교 다항식 (Charlier 또는 Meixner)
- `θ_n = E_p[ψ_n(X)]`: 데이터로부터 추정하는 전개 계수

### 두 가지 전개 체계

| 체계 | 기저 분포 | 직교 다항식 | 매칭 적률 | 적합 상황 |
|------|----------|-----------|----------|----------|
| **Poisson-Charlier (PC)** | Poisson(μ) | Charlier | 평균 (θ₁=0) | 약한 과분산 |
| **Meixner-NB (NBM)** | NegBin(β, c) | Meixner | 평균+분산 (θ₁=θ₂=0) | 강한 과분산 |

---

## 프로젝트 구조

```
skewness-and-kurtosis/
├── src/
│   ├── orthopoly.py        # Charlier/Meixner 정규직교 다항식
│   ├── moments.py          # 하강 팩토리얼 및 중심적률 계산
│   ├── baselines.py        # Poisson/NB 기저 분포 및 파라미터 추정
│   ├── expansions_mom.py   # Linear-tilt PMF 피팅 (PC & NBM)
│   ├── metrics.py          # L1 거리, 로그우도, AIC/BIC
│   └── data_prep.py        # FIFA/보험 데이터 전처리
├── main.py                 # 실제 데이터 분석 (FIFA + 보험)
├── main_simul.py           # 시뮬레이션 연구 (이봉 NB 혼합)
├── main_success_case.py    # PC 수렴성 입증
├── data/
│   ├── results.csv         # FIFA 월드컵 경기 결과
│   └── insurance.csv       # 의료 보험 청구금액
└── result/                 # 생성된 그래프 및 리포트
```

### 모듈 설명

| 모듈 | 역할 | 주요 함수 |
|------|------|----------|
| `orthopoly.py` | 직교다항식 계산 | `get_charlier_psi()`, `get_meixner_psi()` |
| `moments.py` | 적률 계산 | `falling_factorial()`, `central_moments()` |
| `baselines.py` | 기저 분포 | `poisson_baseline()`, `nb_moment_matched_params()` |
| `expansions_mom.py` | 전개 모델 피팅 | `fit_pc_pmf()`, `fit_meixner_pmf()` |
| `metrics.py` | 적합도 평가 | `l1_sum_abs()`, `loglik_from_sample()`, `aic_bic()` |
| `data_prep.py` | 데이터 전처리 | `load_fifa_counts()`, `insurance_bimodal_to_count()` |

---

## 수학적 세부사항

### Charlier 다항식 (논문 Section 2.4)

하강 팩토리얼 전개 (Eq 23):

```
C_n(x; μ) = Σ_{k=0}^{n} C(n,k) (-1)^k (x)_k / μ^k
```

3항 재귀식:

```
μ C_{n+1}(x) = (μ + n - x) C_n(x) - n C_{n-1}(x)
```

Poisson 가중 직교관계:

```
Σ_x w_P(x) C_n(x) C_m(x) = (n! / μ^n) δ_{nm}
```

정규직교 기저: `φ_n(x) = √(μ^n / n!) × C_n(x; μ)`

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

정규직교 기저: `φ_n(x) = M_n(x; β, c) / √h_n`,  `h_n = n! c^{-n} / (β)_n`

### NB 파라미터 추정 (Eq 29)

데이터의 평균 `μ_p`와 분산 `σ²_p`로부터 (과분산 조건: `σ²_p > μ_p`):

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

## 설치

```bash
git clone https://github.com/your-repo/skewness-and-kurtosis.git
cd skewness-and-kurtosis

# uv (권장)
uv sync

# 또는 pip
pip install -e .
```

**요구사항**: Python >= 3.14, NumPy, SciPy, Pandas, Matplotlib, Seaborn

---

## 실행 방법

### 실제 데이터 분석

```bash
python main.py
```

FIFA 월드컵 골 수 및 보험 청구 데이터에 4가지 모델(Poisson, NB, PC, NBM)을 적합합니다. 비교 그래프와 L1 오차 테이블을 `result/` 디렉토리에 생성합니다.

**출력물**:
- `result/plot_fifa_standard.png` - FIFA 4모델 비교 그래프
- `result/plot_ins_standard.png` - 보험 4모델 비교 그래프
- `result/plot_ins_pc_convergence.png` - 보험 PC 차수별 수렴 그래프
- `result/report.txt` - L1 오차 및 파라미터 상세 리포트

### 시뮬레이션 연구

```bash
python main_simul.py
```

이봉(bimodal) 음이항 혼합 데이터(피크: 20, 80 / V/M ≈ 23.5)를 생성하여 극단적 과분산에서의 모델 성능을 비교합니다.

**출력물**:
- `result/plot_simul_standard.png` - 시뮬레이션 4모델 비교 그래프
- `result/report_simul.txt` - 시뮬레이션 분석 리포트

### PC 수렴성 입증

```bash
python main_success_case.py
```

깨끗한 이봉 Poisson 혼합 데이터에서 PC 전개 차수(K = 0, 2, 4, 6, 8) 증가에 따른 L1 오차의 단조 감소를 입증합니다.

**출력물**:
- `result/plot_success_convergence.png` - 차수별 수렴 그래프
- `result/report_success_case.txt` - 수렴 테스트 리포트

---

## 실험 결과

### FIFA 월드컵 골 수 (n=964, 평균=2.82)

| 모델 | L1 거리 | 비고 |
|------|---------|------|
| **NBM** | **0.0670** | 최우수 |
| NB | 0.0714 | |
| PC | 0.0729 | |
| Poisson | 0.1336 | |

NB 기저로 평균과 분산을 동시에 매칭한 후, Meixner 다항식으로 왜도/첨도를 보정하여 최고 적합도를 달성합니다.

### 보험 청구금액 (n=1338, 평균=3.81)

| 모델 | L1 거리 | 비고 |
|------|---------|------|
| **NB** | **0.2690** | 최우수 |
| NBM | 0.3005 | |
| PC | 0.5863 | |
| Poisson | 0.6786 | |

극단적 영과잉(zero-inflation)으로 인해 고차 보정이 음수 확률을 생성하고, 이를 클리핑하는 과정에서 적합도가 저하됩니다. 논문의 feasible set C_K 제약 위반과 관련된 현상입니다.

### 시뮬레이션: 이봉 NB 혼합 (n=4000, V/M=23.5)

| 모델 | L1 거리 | 비고 |
|------|---------|------|
| **NBM** | **0.4122** | 최우수 |
| NB | 0.5073 | |
| Poisson | 1.4779 | |
| PC | 1.5465 | |

극단적 과분산 데이터에서 NBM이 NB 대비 18.8% 개선된 성능을 보입니다.

### PC 수렴성 테스트 (깨끗한 이봉 분포, n=10000, V/M=2.13)

| 차수 | L1 거리 | 감소율 |
|------|---------|--------|
| K=0 (Poisson) | 0.4775 | - |
| K=2 | 0.1439 | -69.9% |
| K=4 | 0.0470 | -67.3% |
| K=6 | 0.0417 | -11.3% |
| K=8 | **0.0234** | -43.9% |

차수 증가에 따른 단조 감소가 Parseval 항등식(Proposition 1)을 실험적으로 확인합니다.

---

## 구현 세부사항

### 계수 추정 방식

전개 계수는 정규직교 다항식의 표본 기대값으로 추정합니다 (Eq 17):

```python
# θ*_n = E_p[ψ_n(X)] 의 표본 추정
psi_at_data = get_charlier_psi(data, mu, K=K)
theta = np.mean(psi_at_data, axis=0)
```

이는 Proposition 2 (Eq 24-27)의 해석적 팩토리얼 적률 공식과 수학적으로 동치이며, 수치적으로 더 안정적입니다.

### 비음수성 처리

Linear-tilt 모델은 `θ ∉ C_K` (feasible set, Eq 20)일 때 `p_θ(x) < 0`을 생성할 수 있습니다. 현재 구현은 음수값을 클리핑하고 재정규화합니다:

```python
def normalize_pmf(p):
    p[p < 0] = 0.0
    return p / p.sum()
```

### 모델 선택 가이드

```
1. V/M 비율 계산: V/M = Var(X) / Mean(X)
   - V/M ≈ 1  → PC 전개 (Poisson 기저)
   - V/M > 1  → NBM 전개 (NB 기저)
   - V/M < 1  → Poisson 또는 Binomial 기저 (미구현)

2. 전개 차수 K 선택
   - K=4: 표준 (왜도 + 첨도 포착)
   - K=6, 8: 복잡한 다봉 분포에 사용
   - L1 오차 감소 추이 확인, 증가 시 과적합 주의

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
