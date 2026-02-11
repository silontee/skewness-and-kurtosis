# 📊 Skewness and Kurtosis: SNP Expansion Framework

본 프로젝트는 **Semi-Nonparametric (SNP)** 확장을 활용하여 복잡한 이산형 확률 분포(Discrete Count Data)의 형태를 정교하게 복원하는 통계 분석 프레임워크입니다. 특히 데이터의 **과분산(Over-dispersion)**과 **이봉성(Bimodality)**이 강한 환경에서 기존 모수적 모델의 한계를 극복하는 데 특화되어 있습니다.

---

## ✨ Key Features

* **Advanced SNP Modeling**: Meixner 다항식 기반 확장(NBM)과 Poisson-Charlier 확장(PC)을 통한 확률 질량 함수(PMF) 복원.
* **High-Order Moment Adjustment**: 왜도(Skewness)와 첨도(Kurtosis)를 직접 제어하여 데이터의 꼬리(Tail)와 비정형 피크를 정확히 추정.
* **Unified Pipeline**: 전처리, 피팅, 시나리오 분석 및 시각화를 아우르는 통합 분석 로직.
* **Professional Reporting**: $L_1$ Discrepancy 랭킹 및 8차 다항식 계수($t_1 \sim t_8$)를 포함한 전문가용 리포트 자동 생성.

---

## 📂 Project Structure

```text
.
├── src/
│   ├── analysis.py      # 통합 분석 및 시각화 전용 엔진 (5종 그래프 생성)
│   ├── expansions_mom.py # 적률법(MoM) 기반 SNP 피팅 및 정규화
│   ├── orthopoly.py     # Meixner & Charlier 직교 다항식 계산 로직
│   ├── baselines.py     # Poisson 및 NB 기초 모델 설정
│   ├── metrics.py       # L1 Discrepancy 및 Empirical PMF 연산
│   └── data_prep.py     # FIFA/보험 데이터 로드 및 전처리 (Binning, Capping)
├── data/                # 분석용 원천 데이터셋 (.csv)
├── result/              # 생성된 시각화 자료 및 통합 리포트
└── main.py              # 데이터 생성 및 전체 프로세스 통합 실행 스크립트
```
---

## 📈 Analysis Scenarios

| Dataset | Type | Characteristics | Key Focus |
| :--- | :--- | :--- | :--- |
| **FIFA** | Real | Heavy Right Tail, $V/M > 1$ | 꼬리 부분의 고차 적률 복원 |
| **Insurance** | Real | Bimodal (Smoker vs Non-smoker) | 이봉성 데이터의 형태 복원 |
| **Simul_Heavy** | Synth | Extreme Over-dispersion ($V/M > 20$) | 극한 환경에서의 모델 안정성(Stress Test) |
| **Simul_Success**| Synth | Clean Bimodal Mixture | 차수 증가에 따른 이론적 수렴성 증명 |

---

## 🚀 Quick Start

본 프로젝트는 `uv` 패키지 매니저를 통한 실행을 권장합니다.

```bash
# 전체 통합 분석 실행 (실제 데이터 2종 + 시뮬레이션 2종)
uv run main.py
```
실행 결과는 result/ 폴더에 다음과 같이 생성됩니다:

* 5개의 분석 그래프: 표준 비교(3개) 및 수렴성 증명(2개)

* 통합 분석 리포트: unified_report.txt

---

## 📊 Result Highlights

* **Standard Comparison**: NBM(Meixner-based) 모델이 과분산이 심한 실제 데이터와 시뮬레이션에서 PC 모델보다 우수한 안정성과 낮은 $L_1$ 오차를 보임을 입증합니다.


* **Convergence Proof**: 'Simul_Success' 시나리오를 통해 다항식 확장 차수($K$)가 증가함에 따라 실제 분포에 지수적으로 수렴하는 과정을 시각화합니다.