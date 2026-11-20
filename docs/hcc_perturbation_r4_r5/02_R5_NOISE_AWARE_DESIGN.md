# 연구 R5: 측정 잡음과 전역 예측 이득을 구분하는 능동적 섭동 선택

**영문 가제:** *Noise-aware Bayesian design for label-efficient HCC perturbation response prediction*  
**버전:** R5-v1.0 / 2026-11-17  
**전용 브랜치:** `research/noise-aware-perturbation-design-r5`  
**과제:** 제한된 수의 HCC 약물 반응을 새로 알 수 있을 때, **무엇을 먼저 관측해야 나머지 약물의 RNA 반응을 더 잘 예측하는가?**

> **주가설:** 불확실한 약물을 무조건 선택하는 대신, **추가 측정으로 줄일 수 있는 예측 불확실성**, 측정 변동의 크기, 같은 batch 안의 정보 중복을 함께 고려하면, 같은 실측 라벨 예산에서 random·diversity·uncertainty 선택보다 미관측 약물의 평균 RNA 예측 오차가 낮아질 것이다.

## 0. 기존 R2와 무엇이 다른가?

- R2: 약물 16개의 반응이 **주어졌을 때**, 그것을 어떻게 사용해야 하는가?
- R5: 16개를 측정할 수 있을 때 **어떤 16개를 선택해야 하는가?**

주 실험에서는 선택 후 데이터를 사용하는 예측기를 모든 정책에 동일하게 고정한다. R2의 새 모델을 기다리거나 반드시 이용하지 않는다. R2와 결합한 실험은 두 연구가 각각 완료된 뒤의 선택사항이며 이번 성공 조건이 아니다.

```text
다른 세포주의 실측 → 기본 반응·표현·측정 변동의 사전정보
                          ↓
새 HCC의 공통 초기 4개 반응
                          ↓
‘나머지 예측에 도움이 될 다음 4개’ 선택
                          ↓
공개 데이터에서 선택한 반응만 공개 → 동일 예측기 갱신
                          ↓
총 8 → 16 → 32개 측정 시 고정 query의 ΔRNA 오차 평가
```

새 wet-lab을 실행하는 계획이 아니다. 이미 측정된 값을 봉인했다가 순차 공개하는 **retrospective experimental-design 평가**다. 실제 신규 실험의 성공이나 환자 치료효과를 주장하지 않는다.

## 1. 선행연구와 출발 가설

| 선행연구 | 확인한 내용 | 이번에 그대로 쓰지 않는 부분 |
|---|---|---|
| BATCHIE [S07] | Bayesian model과 batch acquisition을 이용해 제한된 실험으로 전체 반응 공간을 학습 | 조합약물 viability scalar를 978차원 RNA와 같은 문제로 취급하지 않음 |
| MORPH [S08] | 공개 초기 전문에서 단순 uncertainty/diversity가 random과 비슷했고, learned-loss 기반 선택을 연구 | ‘불확실성 큰 것 선택=항상 효과적’이라는 전제를 두지 않음 |
| Many Needles in a Haystack [S09], ICML2026 | hit를 여러 개 찾는 목표와 optimum 하나를 찾는 목표를 구분 | 본 연구 목표는 hit 수가 아니라 **관측하지 않은 반응의 전체 오차** |
| 프로젝트 B [P02] | mean-offset이 강하고, 적응 16개 외 target 검증 130개를 쓴 사례도 있음 | 실제 라벨 비용을 숨기거나 큰 모델 FT만 비교하지 않음 |

**미해결 지점:** 높은 불확실성이 ‘다음 실험으로 배울 만한 정보’인지 ‘반복해도 흔들릴 측정’인지 구분하지 못하면 예산을 낭비할 수 있다. 또 서로 비슷한 후보 여러 개를 한꺼번에 측정하면 효과가 중복된다. 이것을 RNA 반응의 실제 반복 구조와 source→HCC 전이 조건에서 검증한다.

기본 Bayesian design 공식은 새로운 발명이 아니다. 신규성 후보는 **다변량 반응, source 기반 반복 오차, 실제 숨겨진 라벨 예산, context drift**를 같은 정책에서 다루는 검증 가능한 설계다.

## 2. 데이터 계약과 예측 목표

| 항목 | 정의 |
|---|---|
| 데이터 | 기존 CMap2020 Level3 10µM/24h, 978 landmark. 다른 release·플랫폼 통합 없음 |
| source | 기존 non-target 161개 cell 풀. family 단위 meta-train/meta-validation으로 분리 |
| 대상 | HUH7, JHH5, JHH7. 서로의 반응은 주고받지 않음 |
| query 약물 | 다른 source cell에서 반응이 관측된 약물. 해당 HCC에서는 선택되지 않은 반응 |
| 출력 | 조건 평균 ΔRNA 978개 |
| 라벨 예산 | K=4/8/16/32의 **실제 공개 처리 측정 수**. 추가 target 검증 라벨 0 |
| 독립성 한계 | HCC가 세 계열뿐이며 일부는 plate 하나. 독립 실험실·환자 일반화 아님 |

source의 같은 약물 반응을 사용하는 것은 허용한다. **이 연구를 새 약물 zero-shot이라고 부르지 않는다.**

### 2.1 한 번의 ‘측정’ 비용을 명확히 정의
주 경로는 group마다 metadata/hash 규칙으로 미리 정한 **하나의 treatment well 또는 원자료에서 분리 가능한 최소 처리 관측**을 공개한다.
- `oracle.reveal(group_id)`가 16개 약물의 모든 반복을 무제한 돌려주면 안 된다.
- group의 대표 pert_id·plate/well은 결과값을 보기 전에 고정한다. 다른 alias를 다시 선택해 쉬운 조건으로 바꾸지 않는다.
- 기존 cache가 여러 well을 이미 평균해 분리 불가능하면 원자료 연결을 복구한다. 그래도 불가능하면 예산 이름을 `K aggregated-condition labels`로 바꾸고 실제 원 well 수를 별도 비용으로 보고한다. **K wet-lab 실험으로 표현하지 않는다.**
- 대조군은 사용 가능한 사전 입력이다. 그 측정 비용은 공통 고정 overhead로 따로 기록한다.
- 같은 약물의 재측정 선택은 이번 주 실험에 넣지 않는다. replicate 선택까지 자동 확장하지 않는다.

### 2.2 항상 같은 숨겨진 query 평가
1. target의 eligible drug group을 70% acquisition pool, 30% 봉인 query로 metadata 분할한다.
2. query는 acquisition 후보에 들어가지 않는다. 모든 방법·K에서 동일한 query를 사용한다.
3. query 정답은 알고리즘·하이퍼파라미터 선택·조기종료에 제공하지 않는다. 평가 프로세스에서만 읽는다.
4. 같은 초기 4개를 모든 정책에 제공한다. 이후 한 라운드에 4개씩 선택한다.
5. K=32 pool이 부족한 target은 가능한 K까지만 분석하고 전체 연구를 자동 폐기하지 않는다.
6. 초기 4개/분할 random seed 10개를 기본으로 사용한다. 학습된 source 통계는 고정한다.
7. 기존 공개 자료는 과거 B에서 이미 본 자료다. 새로 획득한 blind cohort인 것처럼 표현하지 않는다.

## 3. 예측기를 고정해 선택 알고리즘의 효과를 분리

### 3.1 기본 반응
source-train에서 약물 p의 cell별 반응을 먼저 평균하고, cell-family를 동등 가중한 값을 f0(p)로 둔다.

가상 target source episode에서는 현재 target family의 모든 약물 반응을 f0·표현·noise prior 계산에서 제외한다. query가 같은 약물 평균으로 되돌아 들어가지 않게 한다.

### 3.2 모든 선택 정책에 동일한 readout
기본 readout은 **f0 + mean-offset + 정규화된 residual Ridge/KRR**다.

`e_i = y_i − f0(p_i)`

`b_S = mean(e_i)`

`yhat_q = f0(p_q) + b_S + K(q,S)[K(S,S)+λI]⁻¹(e_S−b_S)`

- kernel 입력은 ECFP와 source 같은 약물 평균의 source-only 저차원 표현이다.
- λ·출력 scale·kernel 설정은 source validation episode에서 고정한다.
- 각 선택 정책이 얻은 support만 다르고, readout 코드·hyperparameter·추론은 같다.
- mean-offset만 쓰는 두 번째 readout으로 핵심 비교를 한 번 반복한다. 정보 선택의 이득이 특정 readout에만 맞춘 것인지 확인하기 위한 제한된 대조다.
- HCC에서 전체 TranSiGen을 매 라운드 다시 학습할 필요는 없다. 비용을 줄이되 모델링 기여가 selection에 있다는 점을 명확히 한다.

## 4. 첫 진단: uncertainty를 믿어도 되는가?

### D1. source 가상 target에서 random 학습곡선
한 cell-family를 제외한 source로 f0를 만들고, 그 cell의 일부 약물을 공개하며 고정 query 오차를 계산한다. 라벨 수 증가가 실제로 도움이 되는지 확인한다.

### D2. 큰 예측 분산이 다음 오차 감소를 예측하는가?
source episode에서만 후보 하나를 추가했을 때의 실제 query 오차 감소를 측정한다. uncertainty·diversity·예측 오차와의 순위 관계를 분석한다.
- 이 과정에서 query 정답으로 계산한 실제 이득은 **진단 oracle**이다. target acquisition에 직접 사용하지 않는다.
- 효과가 없다는 이유로 HCC 정답을 보고 acquisition 공식을 고르지 않는다.

### D3. 반복 측정의 변동을 예측할 정보가 있는가?
독립 plate가 있는 source의 같은 약물·cell 조건에서 반응 차이를 산출한다. 차이 분산에는 plate 변동도 포함되므로 순수 분자 noise라고 부르지 않는다.
- library/screen metadata·처리 전 DMSO 요약·약물 특징으로 그 변동을 예측할 수 있는지 별도 source-family 검증을 한다.
- target의 아직 선택되지 않은 처리군 QC, 실제 반응 크기, 전체 반복 상관을 acquisition 입력으로 쓰지 않는다.
- 반복이 없는 조건을 noise=0으로 두지 않는다. source prior로 수축하고 ‘불명확’ 상태를 표시한다.
- source 예측 잔차 전체를 측정 noise로 간주하지 않는다. 모델 편향과 noise가 섞이기 때문이다.

## 5. 제안 정책: 실제로 줄일 수 있는 전역 오차를 선택 기준으로

### 5.1 가장 단순한 기준: Bayesian residual regression
source에서 고정한 feature φ(p, X0)에 선형 Bayesian residual 모델을 둔다. intercept를 포함하여 공통 offset의 불확실성도 표현한다. source에서 고정한 response PCA/whitening을 acquisition에만 사용할 수 있으나 최종 예측은 978차원으로 평가한다.

support S가 주어졌을 때 feature parameter covariance를

`C_S = (Λ0 + Σ_i φ_i φ_iᵀ / ν_i)⁻¹`

로 둔다. ν_i는 관측 오차 분산의 source 기반 proxy이며 floor와 shrinkage를 적용한다. 고정 feature의 scalar 출력에서 후보 j의 한 번 측정으로 reference pool R의 latent mean 분산이 줄어드는 정도는

`A(j) = Σ_(u∈R) w_u (φ_uᵀ C_S φ_j)² / (ν_j + φ_jᵀ C_S φ_j)`

이다.

- 분자 `u와 j의 예측이 연결된 정도`가 크면 하나의 측정이 여러 query를 돕는다.
- 분모의 ν_j가 크면 같은 정보를 얻는 데 불리한 측정이라는 뜻이다.
- R은 acquisition pool의 입력 특징만으로 미리 정한 대표 집합이다. 봉인 query의 정답은 사용하지 않는다.
- multi-output에서는 source 고정 response covariance의 가중 trace로 확장한다. scalar 공식을 978개의 독립 유전자 증거로 무분별하게 곱해 posterior를 과신하지 않는다.

**중요한 한계:** kernel·noise·prior가 모두 고정된 Gaussian 모델에서는 covariance update가 관측값 자체에 의존하지 않는다. 이 버전은 **순차 설계**이지만 반응을 보고 선택 규칙이 바뀌는 완전한 outcome-adaptive learning은 아니다. 이 사실을 숨기지 않고 필수 baseline으로 둔다.

### 5.2 outcome-adaptive 확장
주 제안은 위 한 모델 대신 source에서 선정한 소수의 확률모형 가설 h를 사용한다.

예: 화학 구조 kernel 중심 / source 반응 kernel 중심 / 두 정보 결합, 각각 보수적 noise scale. 가설은 최대 6개이며 target 결과를 보고 새로 추가하지 않는다.

공개된 support만으로

`p(h | S) ∝ p(h) p(y_S | X_S, h)`

를 갱신한다. 그러면 같은 4개 약물을 보았더라도 실제 반응이 어떤 형태인지에 따라 다음 선택이 달라질 수 있다.

후보 j에 대해 posterior에서 가상의 관측값 y_j를 생성하고, 그것을 보았다고 가정한 뒤 전역 latent mean 불확실성이 얼마나 줄어드는지 계산한다.

`A_mix(j) = V(S) − E_(y_j~posterior)[V(S ∪ {(j,y_j)})]`

여기서 V는 reference pool의 **측정 noise를 제외한 latent mean posterior variance**의 가중합이다. 가상의 y_j는 모델에서 생성한 계산용 표본이지 실측 정답으로 재학습하는 합성 데이터셋이 아니다.

- 후보 최대 64개 shortlist, fantasy 8개부터 시작한다. 큰 예산의 전수 Monte Carlo가 개선 원인인지 비용 대조를 한다.
- source에서 likelihood temperature/calibration을 정해 978차원의 중복 증거가 posterior를 과신하지 않게 한다.
- 한 라운드의 네 개를 고를 때 먼저 고른 후보의 정보 기여를 조건화해 중복 선택을 줄인다. 단순 top-4 uncertainty와 구분한다.
- batch 안에서 실제 결과를 미리 알고 순서를 바꾸지 않는다. 한 batch 전체를 선택한 뒤 oracle이 네 결과를 공개한다.
- ν의 추정이 불안정하면 동일 분산 모델로 돌아간다. ‘noise-aware’ 기여는 성공했을 때만 주장한다.

### 5.3 구현 범위
- 작은 Bayesian linear/GP posterior와 low-dimensional response basis로 시작한다.
- SciPy/NumPy/PyTorch 또는 검증된 GP 라이브러리를 사용하고, Cholesky/solve·variance non-negativity fixture를 만든다.
- 가장 가까운 외부 기준은 BATCHIE의 PDBAL. 다변량 posterior sample distance를 사용하는 wrapper로 적용한다. 원형 scalar viability와 같다고 주장하지 않는다.
- 기본 R5는 새 foundation model도, LLM 실험 에이전트도 아니다. **동일 예측기를 더 적은 라벨로 유용하게 만드는 데이터 획득 알고리즘 연구**다.

## 6. 필수 비교 정책

모든 정책은 같은 initial4, pool/query, 총 K, readout을 사용한다.

| 정책 | 목적 |
|---|---|
| Random | 반드시 넘어야 할 기본값. 충분한 paired seed 반복 |
| ECFP diversity / farthest-first | 단순 화학적 다양성만으로 충분한가? |
| Predictive total variance | noise와 epistemic uncertainty를 섞으면 실제로 불리한가? |
| Latent-mean variance | 단순 epistemic uncertainty만으로 충분한가? |
| 고정-kernel 전역 variance reduction | outcome-adaptive mixture가 추가로 필요한가? |
| **PDBAL-RNA adapted** | BATCHIE 계열 표준 Bayesian design 대비 추가 가치 |
| **Learned-loss acquisition adapted** | MORPH의 문제의식과 연결되는 오차 예측형 대조. source episode에서만 학습 |
| 제안 noise-aware outcome-adaptive batch design | 주 방법 |

처음에는 Random·diversity·latent variance·고정 전역 감소 네 정책으로 시작한다. 통합 방법이 개발 query에서 이득을 보이면 나머지 강한 대조를 확장한다. 최종 주장은 실행한 비교군 범위로 제한한다.

**Probability-of-Hit는 주 baseline에 넣지 않는다.** 그것은 ‘유용한 hit를 여러 개 발견’하는 다른 목적이다. 큰 RNA 변화=좋은 약물이라는 가정을 만들지 않는다.

## 7. 단계별 작은 실험과 개선 cycle

| 단계 | 질문 | 다음 결정 |
|---|---|---|
| R5-E0 | random에서 K 증가가 실제로 오차를 줄이는가? | 줄이지 않으면 readout/labels·signal을 먼저 점검 |
| R5-E1 | 단순 uncertainty가 random을 넘는가? | 못 넘으면 가설대로 noise·중복 여부를 분석 |
| R5-E2 | 고정 posterior의 전역 위험 감소가 더 좋은가? | 개별 uncertainty 대신 query-space 기여가 필요한지 |
| R5-E3 | source 반복 noise proxy가 미래 관측 변동을 설명하는가? | 안 되면 heteroscedastic 부분은 보류 |
| R5-E4 | outcome-adaptive mixture가 고정 설계보다 좋은가? | 실제 피드백 사용의 추가 기여 |
| R5-E5 | 동결 정책을 HCC3에 적용 | source 개선과 HCC 전이의 차이 보고 |

허용되는 두 개선 cycle:
1. **불확실성–오차 불일치:** source calibration/likelihood tempering 또는 noise shrinkage 중 근거에 맞는 하나를 수정.
2. **동일 약물군 집중·중복:** batch fantasy와 reference coverage를 수정하되 새 목표로 hit discovery를 추가하지 않음.

선택한 약물의 실제 반응이 큰지 작은지는 사후 분석이다. 특정 policy가 잘 보이도록 query를 줄이거나 실패 약물을 삭제하지 않는다.

## 8. 효과가 있을 때의 ablation

- heteroscedastic ν→동일 ν: 측정 변동 반영의 기여.
- outcome 업데이트를 고정 mixture로 대체: 실제 결과 피드백의 기여.
- batch 조정→top-b 일괄 선택: 중복 감소의 기여.
- 같은 support를 mean-offset readout과 residual KRR로 평가: 특정 예측기와의 결합 효과인지.

새로운 표적 유전자형, pathway loss, diffusion을 추가하지 않는다. 정책 자체가 없으면 가설을 기각/보류할 수 있어야 한다.

## 9. 평가

**주지표:** K=4/8/16/32에서 고정 query ΔRMSE의 예산별 곡선 면적을 정규화한 `area_under_error_budget_curve`. 낮을수록 좋다. trapezoidal integration의 K축 간격을 그대로 반영한다.

- K=16 ΔRMSE와 random 대비 차이를 함께 보고한다.
- 비용: treatment 관측 수, control overhead, 추론/선택 wall time을 별도 기록한다.
- ‘random의 K32 성능을 K16으로 달성’은 실제 곡선과 불확실성이 뒷받침될 때만 말한다. 곡선 사이를 임의 외삽하지 않는다.
- ΔPearson, 각자 평균 제거 Pearson, 고정 후보 retrieval은 보조다.
- 10개 시작 seed는 생물학적 target 10개가 아니다. source는 family 단위, HCC 내부는 drug group·실험 block 기준 paired uncertainty를 보고한다.
- HCC3 평균 CI로 환자 HCC 전체에 대한 확률을 주장하지 않는다.
- 반복 plate가 있는 source에서 support/query plate를 분리한 고정 stress test를 둔다. HCC JHH7처럼 plate 하나면 그 검증은 불가능하다고 명시한다.

### 무의미한 성공을 피하는 대조
- 예측만 있고 실제 라벨을 공개하지 않는 simulator에서 누수 여부 검사.
- oracle 파일의 query 값을 교환해도 선택 과정은 변하지 않아야 한다.
- 관측 noise만 큰 후보를 선택하는지, 정보를 주는 후보를 선택하는지 synthetic fixture로 점검한다. 이 fixture 성능을 생물학적 성능처럼 표에 합치지 않는다.

## 10. 성공·중단 기준

- source-family 평가에서 Random·diversity·standard Bayesian design을 넘고 HCC에서 같은 방향이면 데이터 효율의 근거.
- source에서만 개선되면 source experimental-design 결과와 HCC 전이 실패를 함께 보고.
- 고정 variance reduction과 같으면 mixture adaptation은 불필요할 수 있다. 더 단순한 방법을 채택.
- noise proxy를 제거해도 같으면 noise-aware novelty를 주장하지 않는다.
- 두 근거 기반 수정 후에도 random보다 낫지 않으면 선택 정책을 종료하고, 무작위 실험이 충분했던 조건을 정리한다.

## 11. 병렬 운영과 산출물

초기에는 CPU batch linear algebra로 실행하고, learned-loss surrogate가 필요할 때 작은 GPU 학습을 추가한다. R4·R1–R3에 선행 의존성이 없다. 공유 raw만 read-only로 사용한다.

```text
configs/perturbation_design_r5/
manifests/perturbation_design_r5/{pool_query,oracle_costs,source_noise,freeze}.json
src/perturbation_design_r5/{oracle,predictor,posterior,acquisition,replay,evaluate}.py
reports/perturbation_design_r5/{RESULTS,BEGINNER_SUMMARY,DECISIONS,EXPOSURE}.md
tutorials/perturbation_design_r5/
```

튜토리얼은 ‘초기 4개를 공개→다음 4개 선택→정답 공개→나머지 예측’ 한 episode를 작은 데이터에서 따라 하게 한다. 예측 모델의 내부를 모두 이해하지 않아도 연구 질문과 실측 비용을 확인할 수 있어야 한다.

## 12. /goal

```text
/goal
research/noise-aware-perturbation-design-r5에서 R5를 수행하라.
기존 CMap2020 Level3·978genes·동일 dose/time 자료를 재사용하고,
새 HCC에서 적은 관측으로 나머지 관측약물 반응을 예측하는 실험선택 알고리즘을 개발한다.
모든 acquisition에 동일 readout·initial4·pool/query·총 처리관측 예산을 적용한다.
source의 가상 target으로만 정책과 불확실성/측정변동 prior를 개발·선택한다.
HCC에서는 별도 validation 라벨을 보지 말고 K=4/8/16/32개의 실제 공개 측정만 쓴다.
source의 같은 약물 반응은 허용하지만 아직 관측하지 않은 target X1·QC·반복상관은 금지한다.
원래 Gaussian covariance만 쓰면 관측값에 반응하지 않는다는 점을 구분하고,
고정 variance reduction과 outcome-adaptive posterior mixture를 별도 비교한다.
random·diversity·uncertainty·PDBAL-RNA·learned-loss 대조로 검증하고,
mean-offset 또는 residual Ridge의 readout 변화와 acquisition의 효과를 섞지 않는다.
반응을 순차 공개하는 retrospective 평가임을 명시하고 새 wet-lab 성과로 주장하지 않는다.
두 차례 근거 있는 수정 후에도 이득이 없으면 음성 결과까지 정리한다.
기존 branch를 수정하지 말고 코드·cost ledger·선택 sequence·해설·튜토리얼을 전용 경로에 반영하라.
```

**근거:** S07–S13, P01–P05. 전체 제목·원문·확인 범위는 `03_EVIDENCE.md` 참조.
