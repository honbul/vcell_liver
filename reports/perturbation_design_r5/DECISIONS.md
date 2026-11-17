# 연구 R5 — 결정 기록 (DECISIONS)

형식: 관측 → 가설 → 수정 하나 → 예상 차이 → 실제 결과 → 다음 결정. 규칙 파일은 `configs/perturbation_design_r5/protocol.yaml`.
각 항목은 해당 실행 **전에** 작성하고, 결과는 실행 후 같은 항목 아래에 덧붙인다.

## 0. 운영 기록

| 날짜 | 내용 |
|---|---|
| 2026-11-17 | 브랜치 `research/noise-aware-perturbation-design-r5`를 연구 B commit `d67bf68`에서 생성. 계획서 묶음을 `docs/hcc_perturbation_r4_r5/`에 그대로 복사(MANIFEST sha256 일치). |
| 2026-11-17 | 프로토콜 commit `a55bdc2` (발현 값을 읽기 전). |
| 2026-11-17 | 첫 build job 25334는 프로토콜 YAML 구문 오류로 **프로토콜 읽기 단계에서** 멈춤(값을 읽기 전). 블록 인용만 고치고(규칙 변경 없음) job 25335로 재실행. |
| 2026-11-17 | build job 25335: 원자료 경로 오타(`assets/raw` → `assets/raw/cmap2020`)로 instinfo 읽기 전 실패. 경로만 고쳐 재실행. |

## 1. 개발 단계 (job 25367, source dev 16계열만; D3 평가만 meta-val 반복쌍 사용)

**관측**
- D3: 학습 반복쌍 47,193개로 적합. meta-val 6계열(반복쌍 20개 이상) 계열 동등 평균 Spearman(예측 ν, 실측 반복 차이) **0.243**, family bootstrap 95% [0.085, 0.410] → 게이트(≥0.10, 하한>0) **통과**. 계수 중 plate DMSO 분산(log ctl_sd) 1.71이 가장 크다. → heteroscedastic ν 사용.
- readout(무작위 support만 사용): α=0.5, λ=3 선택, dev AUC 0.5494 (mean-offset 0.5548).
- D1(dev, random): main ΔRMSE K=4/8/16/32 = 0.607/0.566/0.544/0.530, mean-offset 0.610/0.570/0.549/0.537. 라벨 수가 늘면 오차가 줄어든다(E0 조건 충족).
- GP 초모수: sb2=0.03(**사전 격자의 최솟값**), s2=1(세 kernel 모두), 동분산 ν=0.5, 이분산 배수 0.5. 이분산 모형의 pooled log-likelihood가 동분산보다 +1367 높다.
- 온도 T=1/32(**사전 격자의 최솟값**). 격자 경계 선택은 그대로 기록하고 격자를 넓히지 않는다(프로토콜 고정).
- D2(K=4, 16계열 평균 Spearman, 후보 추가 시 실제 query 이득과의 관계): 고정 전역 감소 A(j) **+0.23**, 잠재 분산 −0.22, 예측 총분산 −0.37, ν **−0.41**, Tanimoto 거리 −0.02. K=16에서는 모두 약해짐(A +0.03, ν −0.25). 후보 하나 추가의 평균 이득은 K=4 0.0148, K=16 0.0014.
- learned-loss 회귀기: dev 19,984행, 학습 R² 0.79(학습 자료 점수이며 일반화 성능 아님).

**가설(검증 안 함):** 잠재 분산이 큰 후보가 오히려 덜 도움이 되는 것은 ν가 큰 well, 또는 f0가 약한(관측 세포가 적은) 약물에 분산이 몰리기 때문일 수 있다. D2의 상관만으로 원인을 확정하지 않는다.

**다음 결정:** 계획 §6의 순서대로 dev에서 random, diversity, latent_variance, fixed_vr, proposed_nam 5개를 seed 1–10으로 실행한다. proposed_nam의 dev AUC가 random보다 낮을 때만 total_variance, pdbal_rna, learned_loss를 추가한다.

## 2. dev 1단계 (job 25368, `dev_stage1_v1`, 16계열 × seed 10)

**관측 (main readout, AUC, random 대비, family bootstrap 95%)**: diversity −0.0005 [−0.0014, +0.0003]; fixed_vr −0.0014 [−0.0023, −0.0005] (12/16); latent_variance **+0.0130** [+0.0099, +0.0161] (0/16, 더 나쁨); proposed_nam −0.0015 [−0.0038, +0.0006] (12/16). proposed − fixed_vr −0.0001 [−0.0020, +0.0016].
- 효과 크기는 random AUC 0.5494의 약 0.3%다. 선택 시간은 proposed 13초/episode, 나머지 0.1초 미만.
- source 대표 행 중 410개가 2개 이상 well의 평균이라 평균 처리 well 수가 32를 조금 넘는다(cost ledger에 그대로 기록).

**결정:** 프로토콜 staging 규칙(“proposed_nam의 dev AUC가 random보다 낮으면”)의 조건은 점추정으로 충족 → total_variance, pdbal_rna, learned_loss를 dev에서 추가 실행. 동시에 meta-val에서 8개 정책 전체와 반복-분리 stress test를 실행한다. 이 시점에는 dev에서 실패로 볼 근거(무작위보다 나쁨)가 없으므로 수정 cycle은 쓰지 않는다. 축소 실험(ablation)은 meta-val에서 proposed가 random을 넘을 때만 한다.
