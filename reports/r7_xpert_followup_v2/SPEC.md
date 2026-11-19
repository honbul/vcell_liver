# R7 XPert v2 실행 명세 (동결 대상)

근거: private plan 2026-11-19 §4.1–§4.5. 기계 판독용 명세는 `configs/r7_xpert_followup_v2/spec.json`이며, 이 문서와 그 파일이 같은 commit에서 동결된다. 동결 commit SHA는 `configs/r7_xpert_followup_v2/freeze.json`에 기록한다. 동결 전에는 어떤 outer fold의 정답도, R4 test 정답도 채점하지 않는다(pilot은 T 정답을 합성 잡음으로 바꾼 배선 확인만 한다).

## 1. 노출 지위

| 자료 | 이번 트랙에서의 용도 | 지위 |
|---|---|---|
| R4 train 약물(outer 3 fold) | R7-C/R7-L의 공식 평가(T_k) 및 보정 적합(A_k) | 이미 노출된 자료의 후속 개발 |
| R4 validation, is_hcc = 0 (11,602 unit) | 모든 모델의 checkpoint 선택 | 선택 전용, 평가에 쓰지 않음 |
| R4 validation HCC (129 unit) | 정정 run에서 **원래 규칙의 replay**(영향 측정)에만 사용 | 새 선택·보정에 쓰지 않음 |
| R4 test | 정정 재평가, §4.2 6단계 동결 후속 평가 | 노출된 benchmark. 맹검 아님 |

## 2. §4.1 선택 경로 정정

- part 1 감사(`manifests/r7_xpert_followup_v2/selection_population_audit_part1.json`): 원 R7 XPert 6 run(공식 3 + A-2 ECFP 3)은 upstream `d53ff49` 그대로(working copy clean, run별 복사본 blob 동일)였고, h5ad `valid` 11,731행 = R4 validation 전체였으며 그중 HCC 129행(HUH7 13, JHH5 99, JHH7 17; unit 가중 1.1%)이 선택 손실에 들어갔다. 실행 폴더에는 `r7_split_fold_early_stop.pth` 하나뿐이어서 epoch별 재선정은 불가능하다 → **재학습**.
- 정정: 선택 행만 is_hcc = 0 validation으로 바꾸고 unit 가중 upstream loss4 선택기, L0 손실, config, seed 42/43/44, 100 epoch, patience 50(에뮬레이션)을 유지한다. 같은 trajectory에서 원 규칙(HCC 포함)으로 고른 checkpoint도 저장한다(old replay).
- 영향표: old original(v1) / old replay / corrected. test source family-equal ΔRMSE, centred Pearson, tie-corrected MRR, HCC 계열. 판정 기준 없음(정정은 계약 회복과 영향 측정으로 완료).
- 학습 경로 차이(편차): upstream DataLoader 대신 GPU 상주 텐서·seeded randperm 미니배치를 쓰는 wrapper(`src/r7_xpert_followup_v2/xtrain.py`). 모델·config·손실·optimizer·AMP·선택값 계산은 upstream 함수 그대로다. pilot에서 v1 checkpoint의 validation 예측과 logged loss4 재현을 확인한다.

## 3. R7-C — genuine nested OOF affine 보정

- 분할: R4 train 약물 5,802 group을 outer 3 fold(1,934 group씩), 각 A_k 안을 inner 3 fold로 나눔(seed 20261119, `configs/r7_xpert_followup_v2/folds.tsv`, sha 18c38f87…).
- base: L0 recipe, U 선택기(정정된 upstream 규칙). 보조 base: L0-R, L0-P.
- 보정 학습 자료: inner model (k, j)가 A_k의 inner fold j에 낸 OOF 예측. 이 model들은 T_k 약물을 학습·선택·정규화에 쓰지 않는다. checkpoint 선택은 validation non-HCC.
- 팔: C0, Cg, Cs, Cb, Csb. a_c = exp(α₀ + v_c) > 0(scalar), b_c = b₀ + u_c(978 벡터). 전역 prior는 source family 동일 가중 + identity penalty, cell 단계는 cell의 unit별 유전자 평균 제곱 오차의 합 + λ·v_c² + λ·‖u_c‖²/G. 하나의 고정 RMS s로 정규화.
- λ 후보 {0.01, 0.1, 1, 10, 100, identity}, A_k 내부 inner fold 기준 drug-group calibration CV의 source family-equal unit RMSE로 선택. 1e-6 이내 동률이면 가장 강한 shrinkage.
- HCC: HCC train 약물 OOF pair로 HCC별 계수 적합 허용. λ·팔·checkpoint는 non-HCC로만 결정. HCC global-only 진단을 따로 보고. 적합 unit이 없는 cell은 global fallback(flag).
- **1차 대비: Csb − C0 (L0-U), outer fold family-equal source ΔRMSE**, seed 평균 unit 값, drug-group paired bootstrap 95% CI(2,000회, seed 20261117). 개선 확립 = 상한 < 0. 실용 기준 = 추가로 차이 ≤ −0.001. CI가 0 포함 = 미확립. 동등성은 주장하지 않음(사전 margin 없음).
- 보조: Csb−Cs, Csb−Cb, Cg−C0, Cs−C0, Cb−C0, L0-R/L0-P에서 Csb−C0, 보정 후 R−P(L0), HCC, centred Pearson/MRR/RMS ratio, a_c 분포, offset 크기, cell당 drug 수, fallback 수, inner→outer 출력 RMS·평균 이동.
- 불변·fixture: identity 재현, outer label 변조 시 계수·λ·보정 예측 불변, a_c > 0, 후보군(cell × outer fold) 동일, C0 대비 centred Pearson·tie-corrected MRR 차이 < 1e-9(float64).
- 6단계: 전체 train cross-fit OOF(outer model 예측, outer fold로 CV)로 계수를 다시 적합해 정정된 전체 학습 XPert(L0-U)의 test 예측에 적용. 동결 후속 평가로만 보고.

## 4. R7-L — objective × checkpoint 선택 2×2 (raw)

- L0: upstream 복합 loss(가중 0.2/0.003/0.2/1, epoch 70 전후 전환). L1: mean_i mean_g(Δ̂−Δ)²만, 일정. auxiliary head는 구조상 남기고 가중 0.
- 선택기: R = family-equal unit RMSE 최소, P = family-equal unit gene Pearson(centred 아님) 최대, U(보고용). 같은 trajectory에서 고정 100 epoch, 매 epoch 계산, strict 개선 시 저장(동률이면 첫 epoch).
- **1차 대비 5개**: L0: R−P, L1: R−P, R: L1−L0, P: L1−L0, 상호작용 (L1R−L1P)−(L0R−L0P). outer fold family-equal source ΔRMSE. Holm(5) + Bonferroni 동시 CI(1−0.05/5). 실용 기준 0.001.
- horizon 경계 규칙: recipe × selector의 9개 outer trajectory best epoch 중앙값 ≥ 95이면 연장 필요로 판정. 예산이 허락할 때만 연장, 아니면 미실행 기록.
- 보정 layer는 L0 base에만(§4.4 단계 결정). L1 보정은 미실행 기록 대상.

## 5. 집계

metric을 seed별로 계산한 뒤 unit 값을 seed 평균. source = non-HCC unit, R4 family 동일 가중. MRR 후보군 = cell × outer fold(outer 평가) 또는 cell(test). seed나 cell을 독립 반복으로 세지 않는다.

## 6. 비용과 단계 결정 (pilot 측정 후 기록)

pilot(fold 0 / seed 42, 2 epoch; 2026-11-19): epoch당 학습 시간 A30 기준 전체 train 280 s, outer(≈37.5k unit) 185 s, inner(≈25k unit) 120 s, 선택 평가 12–24 s, 최대 GPU 메모리 21 GB. A40 ≈ A30, A100 = 64 s(inner) ≈ 1.86배. trajectory 51개 = A30 환산 약 250 GPU-h. A100 6장 + 기타 2장이면 약 19–22 h로 ~24 h 예산 안이다. 따라서 §4.4 단계 결정(공식 recipe 보정 + raw 2×2 공유, L1 보정은 질문이 생길 때만)대로 51개 전체를 한 번에 실행한다. 실행 순서는 긴 작업 우선(정정 6 → outer 18 → inner 27).

pilot 배선 확인(성능 증거로 쓰지 않음):
- wrapper 동등성: v1 checkpoint(xpr_e100_s42, epoch 96)의 validation 예측이 원 `pred_validation.npy`와 최대 절대차 0.0, 재계산 loss4 0.72355 = 로그 0.724(소수 셋째 자리). treated slot 비의존 확인(< 1e-4).
- CPU 평가 pilot(fold 0 / seed 42, 2 epoch 모델, T 정답 합성 잡음): 9분, 11 GB. 불변 확인 centred Pearson 최대 1.4e-11, MRR 0, tie 수·후보군 동일. outer label 변조 시 계수·λ·보정 예측 동일. identity 재현 0.0. synthetic 회복 fixture 통과(`calibrator_synthetic_fixtures.json`).
- 동결 후 변경 없음 원칙: 이 commit 이후 `src/r7_xpert_followup_v2`, `configs/r7_xpert_followup_v2`(freeze.json 추가 제외), `experiments/r7_xpert_followup_v2`를 바꾸면 GPU worker가 새 trajectory를 집지 않는다.
