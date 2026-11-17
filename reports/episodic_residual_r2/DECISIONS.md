# 연구 R2 — 단계별 결정 기록 (DECISIONS)

각 단계를 **실행하기 전에** 다섯 줄(관측·가설·예측·실험·결정 규칙)을 적고, 실행 뒤에 `결과`와 `결정`을 덧붙인다. 앞에 적은 줄은 고치지 않는다.
수치의 원자료는 `manifests/episodic_residual_r2/`에 있다. "가설"은 검증 전 설명이며 확정 원인이 아니다.

---

## 단계 R2-0 — source 전용 기준선·episode 구성·누출 fixture (2026-11-17, 실행 전 기록)

- **관측:** 연구 B(`d67bf68`)에서 세 HCC 계열 모두 mean-offset 보정이 full fine-tuning보다 시험 ΔRMSE가 낮았다. B의 적응은 k=16개 외에 target 검증 약물(14/130/22 그룹)의 라벨로 학습률·중단 시점을 골랐다. B의 source 분할(145/16)은 세포주 이름 단위였다.
- **가설:** (검증 전) 공통 offset을 뺀 뒤에도 약물별 잔차에 support로부터 예측 가능한 성분이 남아 있을 수 있다.
- **예측:** 가설이 맞다면 source 가상 target에서 ECFP 잔차 Ridge/KRR이 mean-offset보다 query ΔRMSE와 centred 지표에서 낮아야 한다. 맞지 않으면 선택되는 λ가 격자의 가장 큰 쪽으로 가고 점수가 mean-offset과 같아진다.
- **실험:** (1) 161개 source 세포주의 family를 metadata로 정하고 옛 145/16 분할의 family 중복을 확인·최소 수정한다. (2) B의 plate 행을 (세포주, 약물 그룹) 단위로 접는다(source만; HCC 값은 읽지 않는다). (3) episode 생성기와 f0(가상 target family 제외)의 누출 fixture를 합성 자료와 실제 store에서 통과시킨다. (4) 고정 episode에서 무변화·f0·mean-offset·ECFP Ridge·Tanimoto KRR·동일 입력 비학습 KRR을 채점한다. λ는 meta-train dev episode로만 고른다.
- **결정 규칙:** fixture가 하나라도 실패하면 학습으로 넘어가지 않는다. family 중복이 있으면 중복 family 전체를 meta-train으로 옮기는 최소 수정만 한다.

**결과 (R2-0, 2026-11-17):**
- family: 161개 세포주 → 127 family. 옛 분할에서 train·val에 걸친 family는 2개(`XC_POOLS` 29개 혼합 pool, `HIMG001+HIMG002`)였다. 해당 val 세포 5개(XC.P091, XC.P092, XC.P910, XC.P911, HIMG002)를 meta-train으로 옮겼다. meta-train 150 세포주/116 family, meta-val 11 세포주/11 family. XC pool의 구성 세포는 서버 metadata에 없어 **확인하지 못했다**(meta-val family의 세포가 pool에 섞여 있을 가능성은 배제 못 함).
- 단위: source 238,532행 → 78,849 (세포주, 약물 그룹) 단위. HCC 값은 읽지 않았다(`data_contract.json`).
- fixture: 합성 자료와 실제 store 모두 통과(job 24470, `leak_fixtures.json`). 첫 시도(job 24463)는 GPU `index_add`의 합산 순서 차이로 "완전 동일" 비교가 실패했다. f0 표를 CPU float64 고정 순서 합으로 바꿔 재현 가능하게 했다(규칙 변경 아님).
- episode에 쓸 수 있는 세포주(적격 약물 그룹 40개 이상): meta-val 9개(22RV1, BC3C, HCC515, HFL1, MCF10A, MINO, OCILY3, SKMEL5, TC32), meta-train dev 121개(92 family), 학습 sampler 96 family.

**결정:** fixture 통과. R2-1로 진행.

---

## 단계 R2-1 — 진단 D1·D2·D3 (2026-11-17)

> **절차 기록:** D1(closed_form_v1, job 24484)과 D2(d2_v1, job 24472)는 R2-0 작업과 함께 제출되어, 아래 D1·D2의 다섯 줄은 D2 출력을 본 뒤에 적었다. D1의 예측은 R2-0 항목에 미리 적혀 있었다. D3와 그 이후 단계는 실행 전에 적는다.

### D1·D2 (source 가상 target, 닫힌 해법만)

- **관측 (meta-val 9 family, family 동등 평균 query ΔRMSE; `d1_closed_form_summary.tsv`):**

| K | 무변화 | f0 (source 같은 약물 평균) | f0 + mean-offset | ECFP Ridge | Tanimoto KRR | 동일 입력 비학습 KRR |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 0.5369 | 0.5145 | 0.5301 | 0.5298 | 0.5300 | 0.5269 |
| 16 | 0.5369 | 0.5145 | 0.5151 | 0.5148 | 0.5148 | 0.5117 |
| 32 | 0.5369 | 0.5145 | 0.5078 | 0.5069 | 0.5070 | 0.5011 |

  - mean-offset은 K=8에서 f0보다 **나쁘고**(+0.0157, 9개 중 8 family), K=16에서 같으며(+0.0006), K=32에서만 낮다(−0.0067, family bootstrap 구간이 0을 포함). meta-train dev 92 family에서도 같은 방향이다(K=8 +0.0209, K=16 +0.0050, K=32 −0.0054).
  - ECFP 잔차 Ridge/KRR은 mean-offset과 0.001 이내로 같다(K=16 meta-val −0.0003/−0.0002).
  - 동일 입력 비학습 KRR은 mean-offset보다 낮다(meta-val K=8/16/32: −0.0033/−0.0034/−0.0067, 9/9 family). 그러나 이득은 **support와 detection plate를 공유하는 query**에서만 보였다(K=16 meta-val: 공유 0.5079 대 0.5115, 비공유 0.4906 대 0.4906; meta-train dev 비공유에서는 0.5946 대 0.5928로 오히려 높음).
  - D2 (`d2_pair_summary.tsv`, `d2_replicate_summary.tsv`): 세포주별 공통 offset은 f0 잔차 제곱평균의 6.9%다. 같은 약물을 같은 세포주의 다른 plate에서 다시 잰 잔차의 상관은 0.22(다른 약물 −0.006)로, 잔차의 대부분은 반복되지 않는 성분이다. 다른 약물 쌍의 잔차 상관은 같은 plate 0.03, 다른 plate −0.015다. Tanimoto ≥ 0.6인 쌍은 같은 plate에서 0.16, 다른 plate에서 0.02(쌍 2,996개, 전체의 0.05%). f0끼리의 상관이 0.6 이상인 쌍은 다른 plate에서도 0.12다.
- **가설 (검증 전):** (a) mean-offset이 작은 K에서 나쁜 것은 K개 잡음 잔차의 평균이 더하는 분산(대략 잔차 분산/K)이 제거되는 공통 offset(잔차의 약 7%)보다 크기 때문일 수 있다. (b) 비학습 KRR의 이득은 약물별 생물학적 잔차가 아니라 plate별 대조군 편차(같은 plate의 약물이 공유하는 오차)를 대조군 feature로 맞춘 것일 수 있다. (c) f0가 비슷한 약물끼리는 plate가 달라도 잔차가 약하게 닮는다 — 학습된 metric이 쓸 수 있는 유일한 plate 밖 신호 후보다.
- **예측:** (b)가 맞다면 학습된 adapter의 이득도 plate 공유 query에 몰리고, 대조군 feature를 뺀 변형은 mean-offset과 같아진다. (c)가 맞다면 대조군 feature 없이도 plate 비공유 query에서 작은 이득이 남는다.
- **실험:** 계획대로 R2-2(학습 metric)와 R2-3(offset 분리/비분리)을 실행하고, plate 공유/비공유 하위 점수를 모든 표에 싣는다.
- **결정:** 단순 ECFP 잔차 Ridge는 mean-offset과 같다 → 계획서 §3 판정에 따라 "약물 구조만으로는 식별 가능한 약물별 적응 신호가 약하다". learned metric은 한 차례 시험한다(계획 허용). HCC의 기준선 "mean-offset"은 계획대로 유지하되, **f0 자체가 K≤16에서 mean-offset보다 낮거나 같다는 사실 때문에 f0도 반드시 함께 비교한다.**

### D3 (실행 전 기록)

- **관측:** 연구 B에서 JHH7은 update 0이 선택됐고 scratch 일부가 발산했다. B0는 BatchNorm을 쓰고, B의 적응은 `model.train()` 상태(작은 batch의 통계로 running statistics 갱신)였다.
- **가설 (검증 전, 국소):** K개 약물·반복되는 X0로 이뤄진 작은 batch의 통계가 미세조정을 불안정하게 했을 수 있다.
- **예측:** 맞다면 같은 학습률·update 수에서 running statistics를 고정한 쪽의 source 가상 target query ΔRMSE가 family 대부분에서 낮다.
- **실험:** meta-val 9개 세포주, support seed 1, B0 seed 42, K=8/16/32, 학습률 3개 × BN 2가지 × update 6개 지점(`tsg_source.py`). target 검증 라벨 없음. 같은 실행으로 full FT의 source 선택 설정을 K별로 정한다.
- **결정 규칙:** BN 고정이 선택되면 그 설정이 full FT 기준선이 된다. 어느 쪽이든 BatchNorm을 실패의 원인으로 단정하지 않는다. 이 진단을 넓히지 않는다.

---

## 단계 R2-2 — learned metric + 닫힌 해법 adapter, 일반 CNP (실행 전 기록, 2026-11-17)

- **관측:** D1에서 ECFP 잔차 Ridge는 mean-offset과 같고, 같은 입력의 비학습 KRR은 0.003–0.007 낮다(plate 공유 query 한정).
- **가설 (검증 전):** source episode로 학습한 embedding은 비학습 kernel보다 support 잔차를 더 잘 가중할 수 있다.
- **예측:** 맞다면 meta-val K=16에서 adapter < 비학습 KRR(0.5117) < mean-offset(0.5151). 틀리면 adapter ≈ 비학습 KRR 또는 ≈ mean-offset(λ가 커짐).
- **실험:** 초기 통합안 1개 구조(φ: 2층 MLP 128→64, LayerNorm; 입력 = ECFP4 + support 중심화 대조군 32축 + f0 32축 + log 세포 수; 전역 λ 학습). 설정 3개(학습률 1e-4/3e-4/1e-3, seed 42). 일반 CNP도 같은 학습률 3개·같은 episode·같은 손실·같은 예산. 선택은 meta-val K=16.
- **결정 규칙:** 선택된 설정으로 seed 43·44 추가. 발산 실행도 ledger에 남긴다.

### 사후 진단 S1 — f0·offset의 scalar 축소 (실행 전 기록, 2026-11-17)

- **관측:** R2-2 초기 실행 중 일반 CNP(lr 3e-4, seed 42)의 meta-val ΔRMSE가 K=8/16/32에서 0.5038/0.5031/0.5030으로 K에 거의 무관했고, f0(0.5145)보다 낮았다. adapter(lr 3e-4)는 학습 초기에 0.526/0.511/0.500 근처였다. D1에서 mean-offset은 K≤16에서 f0보다 낮지 않았다.
- **가설 (검증 전):** CNP의 이득은 support를 이용한 적응이 아니라, 적은 수의 source 세포주로 만든 f0의 잡음을 줄이는(0 쪽으로 당기는) 효과일 수 있다. 같은 이유로 K개 잔차 평균인 offset도 축소하면 나아질 수 있다.
- **예측:** 맞다면 학습 없이 f0에 n/(n+κ_f), offset에 K/(K+κ_b)를 곱하는 것만으로 meta-val에서 CNP 수준(약 0.503)에 가까워지고 K 의존성이 줄어든다. 틀리면 축소는 f0·mean-offset과 0.001 이내로 같다.
- **실험:** `diag_shrink.py`. κ_f 7개 × κ_b 7개 격자, K별로 meta-train dev에서 선택, meta-val 보고. 학습 모델·HCC 값 없음. target 라벨은 기존 mean-offset과 같은 K개만 사용.
- **결정 규칙:** 이것은 **진단**이며 제안법의 수정(계획서 §8 목록)이 아니고 수정 횟수에 넣지 않는다. 축소 f0·축소 offset이 meta-val에서 기존 기준선보다 낮으면, source에서 고른 κ를 동결해 HCC 표에 **추가 기준선**으로 싣고(계획에 없던 추가임을 표시) 기획서의 판정 기준선(mean-offset)은 바꾸지 않는다.
