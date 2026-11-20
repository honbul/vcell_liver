# R7 XPert v2 결과 — 선택 경로 정정, OOF 출력 보정, 학습 목표 × 선택 기준

근거: private plan 2026-11-19 §4.1–§4.5. 명세: `reports/r7_xpert_followup_v2/SPEC.md`, `configs/r7_xpert_followup_v2/spec.json`. 동결 commit **`16c524a`**(기록 `d1f406c`). 노출 기록: `EXPOSURE.md`. 모든 수치는 동결 뒤 1회 계산했다(예외와 이유는 §6 편차).

요약
- **§4.1 정정:** 원 R7 XPert 6 run(공식 3 + A-2 ECFP 3)은 HCC validation 129 unit이 포함된 validation으로 checkpoint를 골랐다(확인). 재학습한 6 trajectory에서 source-only 규칙과 원 규칙(HCC 포함)은 **6/6 모두 같은 epoch**를 골랐다. 선택 mask의 영향은 이 trajectory들에서 0이다. 정정값(test source ΔRMSE)은 공식 0.53776, ECFP 0.54845다.
- **R7-C:** 세포주별 양의 scale + offset 보정(Csb)은 genuine nested OOF에서 ΔRMSE를 **−0.0092 [−0.0115, −0.0070]** 낮췄다. 사전 기준(CI < 0, ≤ −0.001)을 충족했다. centred Pearson·MRR은 수치적으로 그대로다(최대 차 8e-15, 0). 개선은 주로 scale에서 오고, offset 단독(Cb)은 −0.0004로 작았다.
- **R7-L:** checkpoint 선택 기준(R vs P)의 효과는 두 recipe 모두 미확립이다. ΔMSE-only recipe(L1)는 R 선택에서 **더 나빴다**(+0.0061, Holm 보정 후에도 확립). P 선택에서는 95% CI가 0을 넘지만 사전 지정 familywise 기준에서는 미확립이다. L1은 패턴 지표도 잃었다(centred Pearson −0.03, MRR −0.04). 상호작용은 미확립이다.

---

## 1. §4.1 XPert 자동 checkpoint 선택 경로 정정

**목적.** 원 R7 XPert 비교군의 자동 checkpoint 선택에 HCC validation이 들어갔는지 실제 실행과 연결해 확인하고, 들어갔다면 source-only 선택으로 정정해 영향을 잰다.

**수정 내용.**
- 감사(`selection_population_audit_part1.json`, 최종 `selection_population_audit.json`):
  - 실행에 쓴 upstream working copy는 `d53ff49`였고 clean했다. run별 복사본의 `train_xpert.py`·`utils.py`·`MyDataset.py`·model 파일은 HEAD와 blob이 같았다.
  - h5ad `valid` 11,731행은 R4 validation 전체와 같았다(unit_idx 일치). HCC 129 unit(HUH7 13, JHH5 99, JHH7 17)이 들어갔고, unit 가중으로 1.1%다.
  - 선택값은 upstream `loss4` = Σ(1 − unit ΔPearson)/n_valid였다. 6 run 모두 epoch별 snapshot이 없었다(선택 checkpoint 1개만 있음) → **재학습**.
- 정정: 선택 행만 is_hcc = 0 validation 11,602 unit(1,243 약물 그룹, 110 family; hash `3271b832…`)으로 바꿨다. unit 가중 upstream 선택기, L0 손실, config, seed 42/43/44, 100 epoch, patience 50(에뮬레이션)은 유지했다.
- 같은 trajectory가 원 규칙(HCC 포함 전체 validation)으로 고른 checkpoint도 저장했다(old replay). 따라서 mask 효과만 분리된다.
- 학습 wrapper(`xtrain.py`)는 upstream 모델·config·손실·optimizer·AMP·선택값 함수를 그대로 쓴다. 데이터 공급만 GPU 상주로 바꿨다. v1 checkpoint의 validation 예측을 최대 절대차 0.0으로 재현했고, loss4는 0.72355(로그 0.724)였다.

**실험 입력/정답.** 입력은 X0, UniMol + HG(공식) 또는 Morgan + 0 HG(ECFP), cell index다. 정답은 R4 Δ다. 평가는 R4 test(노출된 benchmark; 정정 재평가)다.

**비교 조건.** old original(v1 checkpoint), old replay(새 trajectory, 원 규칙), corrected(같은 trajectory, source-only 규칙). seed별로 계산한 뒤 unit 값을 평균했다. drug-group paired bootstrap 2,000회.

**결과.**

| 변형 | seed | 원 선택 epoch | replay epoch | 정정 epoch |
|---|---|---|---|---|
| 공식 | 42 / 43 / 44 | 96 / 84 / 76 | 94 / 97 / 87 | 94 / 97 / 87 |
| ECFP | 42 / 43 / 44 | 96 / 89 / 75 | 41 / 97 / 99 | 41 / 97 / 99 |

test, source family-equal(106 family), 3 seed 평균:

| 모델 | ΔRMSE | centred Pearson | MRR | HCC 평균 ΔRMSE | RMS 비 |
|---|---|---|---|---|---|
| 공식 old original (v1 보고값) | 0.53604 | 0.183 | 0.352 | 0.60833 | 0.352 |
| 공식 old replay = corrected | **0.53776** | 0.191 | 0.361 | 0.60440 | 0.387 |
| ECFP old original (v1 A-2) | 0.54385 | 0.136 | 0.322 | 0.61210 | 0.346 |
| ECFP old replay = corrected | **0.54845** | 0.127 | 0.308 | 0.61268 | 0.345 |

| 대비 (ΔRMSE) | 공식 | ECFP |
|---|---|---|
| corrected − old replay (mask 효과) | 0 (모든 unit 동일) | 0 |
| corrected − old original | +0.00173 [−0.00306, +0.00751] | +0.00460 [+0.00134, +0.00785] |
| 같은 대비, HCC 평균 | −0.0039 [−0.0100, +0.0021] | +0.0006 [−0.0053, +0.0059] |

선택 집단(validation source) 점수는 공식 v1 0.54802 → 정정 0.54930, ECFP 0.55824 → 0.55496이다(선택에 쓴 자료이므로 서술용).

**의미.**
- 관측: HCC 129 unit이 선택에 들어간 것은 사실이다. 그러나 같은 학습 곡선에서 HCC 포함 여부는 고른 epoch를 바꾸지 않았다(6/6).
- 관측: v1과 정정값의 차이(공식 +0.0017, CI가 0 포함; ECFP +0.0046, CI가 0 미포함)는 mask가 아니라 **재학습 변동**이다. 원 run과 새 trajectory는 GPU, 미니배치 순서, 데이터 공급 경로가 다르다.
- 가설(미검증): ECFP 차이는 seed 42 trajectory(epoch 41에서 조기 종료 에뮬레이션)의 변동이 클 수 있다.
- 정정은 완료다. 원 R7 결론(B11 미채택, XPert ΔRMSE는 같은 범위, 패턴 지표 우위)의 방향을 바꾸는 근거는 없다. 원 보고서에는 덧붙임 메모를 남겼다(`ERRATUM_R7_V1_XPERT.md`).

**다음 행동.** 이 run들의 정정된 checkpoint(L0-U, source-only)를 R7-C 6단계의 최종 base로 썼다. 추가 정정 작업은 없다.

---

## 2. R7-C — genuine nested OOF affine 보정 (공식 recipe, U 선택)

**목적.** XPert의 높은 패턴 지표와 비슷한 RMSE의 차이 일부가 예측 크기나 세포주 공통 이동에서 오는지 시험한다. 이를 위해 패턴을 바꾸지 않는 세포주별 보정 z = a_c·p + b_c(a_c > 0 scalar, b_c 978 벡터)를 쓴다.

**수정 내용.** 신규 `fit_calibrator.py`, `evaluate_oof.py`를 추가했다. 2단계 적합(family 동일 가중 global prior + cell 단계 shrinkage), 하나의 고정 RMS로 정규화, λ ∈ {0.01, 0.1, 1, 10, 100, identity}, 1e-6 동률이면 가장 강한 shrinkage를 쓴다.

**실험 입력/정답.**
- R4 train 약물 5,802 그룹을 outer 3 fold로 나눴다. A_k 안에서 inner 3 fold로 다시 나눴다. inner model 27개가 A_k의 OOF 예측을 만들었고(T_k 약물은 학습·선택·정규화 어디에도 쓰지 않음), outer model 9개가 T_k를 예측했다.
- 모든 checkpoint 선택은 validation non-HCC로 했다.
- 보정 계수와 λ는 A_k OOF와 inner-fold calibration CV로만 정했다. T_k 정답은 채점에만 썼다.
- 평가 unit: source 55,846(127 family, 5,802 그룹), HCC 585(HUH7 46, JHH5 455, JHH7 84).

**비교 조건.** C0 / Cg / Cs / Cb / Csb, 같은 원시 예측·후보군(cell × outer fold)·seed. **1차 대비 Csb − C0**(L0-U).

**결과.** outer fold, source family-equal ΔRMSE(3 seed 평균; C0 = 0.61297):

| 대비 | 차이 | 95% CI | 판정 |
|---|---|---|---|
| **Csb − C0 (1차)** | **−0.00920** | **[−0.01148, −0.00697]** | **개선 확립, 실용 기준(≤ −0.001) 충족** |
| Cs − C0 | −0.00465 | [−0.00713, −0.00241] | 개선 |
| Cg − C0 | −0.00367 | [−0.00589, −0.00141] | 개선 |
| Cb − C0 | −0.00036 | [−0.00055, −0.00020] | 작은 개선(실용 기준 미달) |
| Csb − Cs | −0.00455 | [−0.00557, −0.00320] | 결합 > scale 단독 |
| Csb − Cb | −0.00884 | [−0.01105, −0.00660] | 결합 > offset 단독 |
| L0-R base: Csb − C0 | −0.00625 | [−0.00799, −0.00387] | 개선 |
| L0-P base: Csb − C0 | −0.00889 | [−0.01107, −0.00654] | 개선 |

- seed별 Csb − C0: −0.0123, −0.0080, −0.0073(모두 음수).
- HCC(같은 세포주의 새 약물): 평균 −0.0140 [−0.0178, −0.0101]. HUH7 −0.0246, JHH5 −0.0078, JHH7 −0.0096. "HCC global-only" 진단(HCC 정답 미사용)에서는 0.65976 vs C0 0.66001(−0.0003)이었다. HCC 개선은 거의 전부 HCC train 약물로 적합한 HCC별 계수에서 왔다.
- 패턴 불변: C0 대비 centred Pearson 최대 차 8.1e-15, tie-corrected MRR 차 0, tie 수·후보군 동일. raw Pearson은 바뀐다(불변 주장 대상 아님).
- 계수: 예측 RMS 비가 C0 0.400 → Csb 0.360, Cs 0.300이었다(scale < 1, 예측 진폭 축소). 선택 λ는 Csb에서 10(3)·100(6), Cb에서 identity 7/9였다. fallback 0. 모든 a_c > 0.
- inner→outer 이동: 원시 예측 RMS가 inner OOF(A_k) 0.24–0.28, outer(T_k) 0.20–0.31로 fold마다 달랐다(`inner_outer_shift.tsv`).
- 6단계(동결 후속 평가, R4 test): 전체 train cross-fit OOF 계수를 정정된 full model(L0-U) 예측에 적용했다. Csb − C0 = −0.0090 [−0.0125, −0.0052](C0 0.53776 → Csb 0.52873). HCC 평균은 −0.0101 [−0.0213, +0.0010]이었다. 불변 차 ≤ 3.4e-15, MRR 0.
- 불변·fixture: identity 재현 0.0, outer label 변조 시 계수·λ·보정 예측 동일, synthetic 회복 통과.

**의미.**
- 관측: 세포주별 scale + offset 보정만으로 패턴을 바꾸지 않고 ΔRMSE가 사전 기준 이상 낮아졌다. 개선의 대부분은 예측 진폭을 줄이는 scale에서 왔다. offset 단독은 작았고, scale과 결합할 때 추가로 −0.0046이 생겼다.
- 해석 범위: 이것은 크기·공통 이동 보정의 이득이다. 새로운 약물 패턴 학습이 아니다(plan §4.2 결과표 "RMSE 개선, 패턴 불변").
- 관측: R4 test 후속 평가에서도 같은 크기(−0.0090)였다. 이 값은 노출된 benchmark의 후속 평가이며 맹검이 아니다.
- 주의: 다른 R7 모델(B11, chem_only, Ridge)에는 같은 보정을 적용하지 않았다. 보정 XPert의 test 0.5287을 v1 표의 미보정 모델과 직접 비교해 우위를 주장하지 않는다.

**다음 행동.** 낮은 용량의 보정기(Csb)를 XPert 출력 후처리로 유지할 근거가 있다. 같은 보정을 비교 모델에도 적용한 공정 비교, 또는 독립 자료 확인은 사용자 결정 사항이다(§7).

---

## 3. R7-L — 학습 objective × checkpoint 선택 2×2 (raw)

**목적.** XPert의 ΔRMSE와 패턴의 차이가 학습 recipe(복합 L0 vs ΔMSE-only L1) 또는 checkpoint 선택 목적(R: family RMSE vs P: family gene Pearson)에서 오는지 시험한다.

**수정 내용.** `xtrain.py`의 L1 recipe(mean_i mean_g(Δ̂−Δ)², 보조 head 가중 0)와 매 epoch R/P/U 선택을 추가했다. 고정 100 epoch에서 각각의 최고 checkpoint를 저장했다.

**실험 입력/정답.** R7-C와 같은 outer fold·입력·seed다. trajectory 18개(L0 9, L1 9)이고, 평가 unit은 R7-C와 같다.

**비교 조건.** 1차 대비 5개, Holm(5) + Bonferroni 동시 CI. 실용 기준 0.001.

**결과.** 수준(source ΔRMSE): L0-R 0.61027, L0-P 0.61205, L1-R 0.61633, L1-P 0.61670(보고용 L0-U 0.61297, L1-U 0.61585).

| 1차 대비 | 차이 | 95% CI | 동시 CI | Holm p | 판정(사전 기준) |
|---|---|---|---|---|---|
| L0: R − P | −0.00178 | [−0.00430, +0.00038] | [−0.0051, +0.0012] | 0.28 | 미확립 |
| L1: R − P | −0.00037 | [−0.00305, +0.00185] | [−0.0039, +0.0027] | 0.69 | 미확립 |
| R: L1 − L0 | **+0.00606** | [+0.00156, +0.01132] | [+0.0007, +0.0129] | 0.010 | **L1이 나쁨(확립)** |
| P: L1 − L0 | +0.00465 | [+0.00035, +0.00895] | [−0.0008, +0.0099] | 0.14 | familywise 기준 미확립(95% CI만 0 초과) |
| 상호작용 | +0.00142 | [−0.00139, +0.00430] | [−0.0024, +0.0052] | 0.69 | 미확립 |

- 패턴: L1은 같은 selector에서 centred Pearson −0.031(R)/−0.034(P), MRR −0.040/−0.041, 예측 RMS −0.06/−0.08이었다. R − P는 패턴 차이가 미확립(CI가 0 포함)이고, R이 예측 RMS를 0.039 줄였다.
- 보고용: L0에서 R − U −0.0027 [−0.0053, −0.0002], P − U −0.0009 [−0.0024, +0.0007].
- 보정 후(L0, Csb): R − P +0.0009 [−0.0012, +0.0031]. raw의 R 쪽 점추정 우위는 보정 후 남지 않았다.
- seed별 R: L1 − L0 = +0.0092, +0.0008, +0.0082(모두 양수).
- best epoch 중앙값: L0-U 90, L0-R 81, L0-P 90, L1-U 97, L1-R 90, L1-P 97. **horizon 경계 규칙(중앙값 ≥ 95)이 L1-P·L1-U에서 충족됐다** → 연장 필요로 판정했으나 미실행(§5).

**의미.**
- 관측: 이 backbone·입력·100 epoch에서 ΔMSE-only로 바꾼 recipe는 RMSE를 낮추지 못했다. R 선택에서는 확립된 악화, P 선택에서는 familywise 기준 미확립의 악화 방향이다. 패턴 지표도 잃었다.
- 관측: 선택 기준만 바꾸는 효과(R vs P)는 확립되지 않았다.
- 가설(미검증): raw R − P의 작은 점추정 차이는 R 선택이 예측 진폭이 작은 epoch를 고르는 데서 왔을 수 있다(R의 RMS가 작고, 보정 후 차이가 사라짐).
- 가설(미검증): L1의 best epoch가 경계에 몰린 것(L1-P 중앙값 97)은 L1이 100 epoch 안에 수렴하지 않았을 가능성을 뜻한다. 따라서 "L1이 나쁘다"는 결론은 **100 epoch horizon에 한정**된다.
- plan §4.3 해석 규칙대로, "단순 objective 불일치" 가설은 낮춘다. XPert의 RMSE 쪽 개선은 학습 objective보다 출력 크기 보정(R7-C)에서 더 직접적으로 얻어졌다.

**다음 행동.** L1 연장(같은 규칙으로 모든 해당 팔)은 미실행 기록으로 남긴다(§5). L1 recipe의 보정 layer는 §4.4 단계 결정대로 실행하지 않았다.

---

## 4. §4.5 진입 조건

`ENTRY_CONDITIONS_4_5.md`에 기록했다. 새 HCC 세포주·환자 전이(cell-OOD)와 graph 확대는 모두 미실행이다. 필요한 자료·계약은 해당 문서에 적었다.

## 5. 미실행 기록

| 항목 | 상태 | 이유 | 해소 조건 |
|---|---|---|---|
| L1 horizon 연장 | 미실행 | 사전 규칙(best epoch 중앙값 ≥ 95)이 L1-P·L1-U에서 충족됐다. 그러나 마지막 epoch 상태를 저장하지 않아 outer trajectory 18개를 모두 다시 학습해야 한다(약 80 A100-GPU-h). 예산과 사용자 GPU 축소 지시(g4 6장) 때문에 실행하지 않았다. | 사용자 승인, 더 긴 고정 horizon(예: 200 epoch)으로 L0·L1 outer 18개 재학습, 새 version 기록 |
| L1 recipe의 nested 보정 | 미실행 | §4.4 단계 결정(공식 recipe 보정 우선). raw L1이 L0보다 나빠 답할 질문이 약하다. | L1 연장 후 L1이 L0와 비슷해질 때 |
| 다른 R7 모델(B11, chem_only, Ridge)에 같은 보정 적용 | 미실행 | plan 범위 밖 | 사용자 결정 |
| §4.5 새 HCC·graph | 미실행 | 자료·계약 부재 | `ENTRY_CONDITIONS_4_5.md` |
| XPert UniMol vs HG 기여 분리, 2500 epoch | 미실행 | 원 R7 기록 유지, 이번 plan 범위 밖 | — |

## 6. 편차와 사건

1. **학습 wrapper(D1).** upstream DataLoader 대신 GPU 상주 텐서와 seeded randperm 미니배치를 썼다. v1 checkpoint 예측 재현(최대 차 0.0)과 loss4 재현으로 forward·선택 계산의 동등성을 확인했다. 학습 경로는 bitwise로 같지 않다. 이 때문에 v1과 새 run의 차이에는 재학습 변동이 들어간다(§1).
2. **fold model 선택 집합(D2).** plan의 "A 내부 non-HCC development unit"을 R4 validation non-HCC로 구현했다. 이 집합은 A_k, T_k와 모두 겹치지 않는다. 모든 model이 같은 선택 집합을 써서 A_k 전체를 학습에 쓸 수 있다.
3. **GPU worker 취소와 재실행(D3).** 2026-11-19 ~14:30에 coordinator가 사용자 지시로 g4 외 worker 12개를 취소했다(GPU cap 18 → g4 6장). 이로 인해 trajectory 12개가 중단됐다: full_official s42/s43(epoch 28–29, A30), outer 10개(1 epoch). 처리 내용:
   - 중단 run을 `runs/aborted/`에 보존하고 claim을 해제했다.
   - 재시도 queue(`queue_retry1.txt`, LPT 순서)를 commit `bca1d0a`로 추가했다. 학습 코드는 동결과 같다(추가 파일만).
   - g4 worker 6개가 기존 worker 하나씩 끝난 뒤 시작하도록 해 동시 GPU는 6장 이하였다.
   - 최종 trajectory 51개는 **모두 A100(g4)**에서 학습됐다. 기록: `aborted_runs.tsv`, `gpu_workers.tsv`.
4. **R 선택기 NaN 버그(D4, 동결 후 코드 수정).** outer_L0_k2_s42가 epoch 31부터 NaN으로 발산했다. family_mean의 NaN-skipping 합 때문에 그 epoch의 R이 0.0으로 계산돼 NaN checkpoint가 R best로 저장됐다(U·P는 발산 전 epoch를 골라 정상).
   - 발견 경위: OOF 평가기의 유한성 assert가 실패해 outer 점수가 하나도 쓰이기 전에 멈췄다.
   - 다른 50개 trajectory에는 비유한 epoch가 없음을 확인했다.
   - commit `1f95ead`로 R/P가 비유한 점수를 best로 고르지 않게 고쳤다(U는 upstream NaN 의미 그대로). 해당 trajectory 하나를 처음부터 다시 학습했다(발산 없음; U 76, R 72, P 88).
5. **final eval 2회 실행(D5).** 첫 실행(job 27798)은 6단계가 위 재학습 trajectory에 의존하므로 6단계 전에 취소했다. 그 전에 쓰인 정정 영향 파일은 재실행 결과와 비교했다. 점추정과 나머지 파일은 byte 동일했고, 일부 CI 경계만 1e-9 수준으로 달랐다(float32 bootstrap 행렬곱의 다중 스레드 비결정성). 보존 위치: `runs/eval_final_attempt1_job27798/`.
6. **MRR 정의.** 보고 MRR은 tie-corrected 기대값이다. 이번 예측에는 tie가 없어 v1 legacy midrank 값과 같았다(v1 0.352 재현).
7. **ECFP(A-2) 정정 포함.** 영향받은 XPert 비교군이라 같은 방식으로 정정했다.

## 7. 사용자 결정이 필요한 것

1. L1 horizon 연장(사전 규칙이 연장 필요로 판정; 약 80 A100-GPU-h)을 실행할지.
2. Csb 보정을 B11·chem_only·Ridge 등 비교 모델에도 같은 genuine OOF 절차로 적용해 공정 비교할지. 지금은 XPert만 보정됐다.
3. 보정 XPert를 기존 benchmark 밖의 독립 자료에서 확인할지.

## 8. 자원 기록

| 항목 | 값 |
|---|---|
| GPU, 본 실행(trajectory 51개, 모두 A100) | 134.3 GPU-h (`run_ledger_totals.json`) |
| GPU, pilot·timing probe | 1.0 GPU-h |
| GPU, 중단·폐기 run | 약 8.7 GPU-h (발산 run 2.9 A100-h, 취소된 full 2개 약 4.8 A30-h, 취소된 outer 10개 약 1 GPU-h) |
| CPU | 약 12 core-h(감사·fold·pilot·평가; 가장 큰 작업은 OOF 평가 1h20m, 12 GB) |
| 디스크(서버 assets) | 15.4 GB (예산 30 GB) |
| wall-clock | GPU 2026-11-19 12:06 → 2026-11-20 14:44(재학습 포함), 평가 종료 16:06 |
| 동시 GPU | 8(12:06–14:25) → 최대 18(~14:25–14:30, 취소됨) → 6 이하(g4, 이후 전 기간) |

## 9. 산출물

- `manifests/r7_xpert_followup_v2/selection_population_audit.json`(+ `_part1`), `orig_xpert_logged_curves.tsv`
- `final/`: `correction_*`(정정 영향표, 선택 epoch, 선택 집단 점수), `step6_*`(6단계 test), `final_eval_record.json`
- `oof/`: `r7c_contrasts.tsv`, `r7l_contrasts.tsv`, `primary_by_seed.tsv`, `per_seed_summary.tsv`, `seedmean_summary.tsv`, `calibration_tuning.tsv`, `calibration_coefficients.tsv`, `invariance_audit.tsv`, `inner_outer_shift.tsv`, `best_epochs.tsv`, `oof_eval_record.json`
- `calibrator_synthetic_fixtures.json`, `pilot/`(배선 확인, 합성 T 정답), `run_ledger.tsv`, `run_ledger_totals.json`, `aborted_runs.tsv`, `gpu_workers.tsv`
- 서버 전용(커밋 안 함): checkpoint·예측 `assets/runs/traj/<id>/`, seed 평균 unit 지표 `assets/runs/eval/oof_v1/`
