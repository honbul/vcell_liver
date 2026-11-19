# R9 v2 — 외부 정규화 정정 · matched 외부 모델 대비 · prototype 실패 원인 분리

branch `research/r9-normalization-prototype-v2` (base R9 tip `1d863eb`). 근거: private plan 2026-11-19 §6, §2, §8, §9.
서버 `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r9_norm_proto_v2` (v1 자산은 `vcell_liver_r9_rare_state`에서 읽기 전용 symlink).
Freeze commit(R9-C1·E1/E2 실행 명세, 외부/source 점수 산출 전): `5c9340e`.

## 한눈에

| 패키지 | 결과 | 판정 |
|---|---|---|
| R9-C2 matched seed 0 외부 모델 대비 | 60/60 key 일치. 방향이 바뀐 3행: Fib MLP_donor−ProtoCloud −0.048 [−0.099, +0.001], B A11−scPoli −0.041 [−0.123, +0.044], Fib A11−scPoli +0.150 [−0.003, +0.315] → 모두 우열 불확실 | 문구 정정 완료(v1 문서에 노트) |
| R9-C1 외부 정규화 정정 | 변환 비트 재현(T1), 84 head 중 81 replay 통과(T2), 2,100 key×3 version 완전. LR_donor Corrected−Replay: nat −0.0006 [−0.0011, −0.0002], B −0.0010 [−0.0025, +0.0004], Endo +0.0009 [−0.0042, +0.0081], **Fib −0.0197 [−0.0328, −0.0084]**; fine coverage 0.2–7.8%p 감소 | 정정 완료; 선정 LR_donor 유지 |
| R9-E1 K × 대각 분산 | (a) +0.027 미확립, (b) +0.013 미확립(target별 반대 방향), **(c) pooled가 성분별 고정 분산보다 0.014 유리(확립)** | 원인 후보 = 희소 성분 분산 추정; 확정 아님 |
| R9-E2 fallback vs capacity | supported_stop·matched_kstar가 A11보다 +0.023·+0.025(확립), 서로 ±0.01 안 동등 | 이득은 용량(target 성분 수) 감소로 설명됨; donor-supported partition의 추가 가치 미관측 |
| 강한 기준선 | 새 arm 최선도 LR_donor와 차이 미확립, MLP_donor보다 −0.06 이상 낮음 | 새 prototype 채택 근거 없음 |
| §6.4 자연 희귀 자료 | 조건만 기록 | 미실행(자료 없음) |


## R9-C2 — seed 0 matched 외부 모델 대비 (집계만, 새 학습 없음)

- **목적**: v1 `a2_external_model_contrasts.tsv`는 우리 head의 3-seed donor 평균에서 외부 모델(scPoli/ProtoCloud)의 seed 0을 뺐다. `r9tasks.select`의 seed는 sparse source donor·target 세포를 바꾸므로 두 쪽의 학습 표본이 다르다.
- **수정 내용**: 두 방법 모두 seed 0, 같은 condition·held-out donor로 짝지음(`scripts/r9/v2/c2_matched_seed0.py`). 짝 키 확인표 `a2_external_model_contrasts_matched_seed0_keys.tsv`: 60 key(scPoli 40 + ProtoCloud 20) 모두 design(선택 donor)·n_train·시험 cell ID hash·truth·seed 일치(60/60).
- **실험 입력/정답**: A2 바깥 LOPO(GSE149614 10 donor; 개발 증거, 전원 이전 노출) primary AUPRC(nat = 전 label donor-macro, sparse = target AUPRC). donor paired bootstrap 10,000회(seed 0) 95% CI.
- **결과** (`reports/r9_rare_state/a2_external_model_contrasts_matched_seed0.tsv`; v1 비정합 값 병기)

| 조건 | 대비 | matched seed 0 차이 [95% CI] | v1(3-seed 평균 − seed 0) | 문구 |
|---|---|---|---|---|
| sparse_Fib | A11−ProtoCloud | −0.143 [−0.238, −0.050] | −0.132 [−0.222, −0.051] | A11 열세 유지 |
| sparse_Fib | LR_donor−ProtoCloud | −0.111 [−0.162, −0.056] | −0.122 [−0.182, −0.063] | LR 열세 유지 |
| sparse_Fib | MLP_donor−ProtoCloud | −0.048 [−0.099, +0.001] | −0.054 [−0.087, −0.023] | **우열 불확실(정정)** |
| sparse_B | A11−scPoli | −0.041 [−0.123, +0.044] | −0.045 [−0.096, −0.008] | **우열 불확실(정정)** |
| sparse_Fib | A11−scPoli | +0.150 [−0.003, +0.315] | +0.161 [+0.006, +0.325] | **우열 불확실(정정)** |
| sparse_Endo | A11−scPoli | +0.144 [+0.048, +0.259] | +0.125 [+0.034, +0.235] | A11 우세 유지 |
| nat | A11−scPoli | +0.012 [−0.006, +0.024] | +0.011 [−0.006, +0.024] | 우열 불확실(변화 없음) |
| nat | A11−ProtoCloud | +0.022 [+0.017, +0.029] | +0.022 [+0.016, +0.029] | A11 우세 유지 |

  나머지 10행(LR_donor/MLP_donor × scPoli, nat ProtoCloud)은 방향 변화 없음. ProtoCloud의 sparse_B/Endo는 v1에서 미실행이며 채우지 않았다.
- **의미**: "ProtoCloud가 sparse_Fib에서 모든 같은-표현 head보다 높다"는 MLP_donor와의 matched 비교에서 확립되지 않는다. "scPoli가 sparse_B에서 A11보다 높다"와 "A11이 sparse_Fib에서 scPoli보다 높다"도 확립되지 않는다. 결론 "R9 방법(A11)이 두 외부 모델을 일관되게 이겼다고 할 수 없다"는 유지된다. v1 문서(RESULTS/CLAIMS/BEGINNER_SUMMARY/DECISIONS)에는 원문을 지우지 않고 정정 노트를 추가했다.
- **다음 행동**: 외부 모델 seed 1/2 추가는 정정의 선행 조건이 아니며 실행하지 않았다(미실행 기록 참고).

## R9-C1 — 외부(GSE151530) 정규화 정정 (`reports/r9_rare_state/a4_external_norm_v2/`)

**지위**: 이미 한 번 채점된 외부 평가의 구현 정정(blind/독립 test 아님). 명세 `RUN_SPEC.md`·`configs/r9/a4_external_norm_v2.json`, freeze `5c9340e`(T1/T2 허용오차·calibration 정책·주 대비를 외부 점수 산출 전에 고정).

- **목적**: source와 외부의 정규화 분모를 맞춘다. v1 `a1_repr.py`는 source를 공통 unique gene 전체(18,618개) 합으로 CP10k+log1p 한 뒤 HVG를 골랐지만, 외부는 HVG 2,000개로 먼저 자른 뒤 그 합으로 정규화했다.
- **수정 내용** (`scripts/r9/v2/c1_transform.py`, `c1_a4_refit.py`, `c1_a4_eval.py`): 외부 raw count를 source와 같은 공통 unique gene 목록·순서(18,618; 중복 포함 교집합 18,624는 v1 RESULTS 수치, 외부 중복 symbol 6개 제외)로 정렬 → 그 합으로 CP10k+log1p → source HVG → source scaler/PCA50/PC scaler. 외부에서 적합하는 것은 없다. v1은 변환 객체·head checkpoint를 저장하지 않았으므로 변환을 같은 코드로 재적합하고 84 head를 재적합했다(새 bundle과 checkpoint는 서버 `assets/cache/r9_repr_external_norm_v2/`, `assets/runs/a4_external_norm_v2/`).
- **구현 검증** (`implementation_checks.tsv`, 10/10 통과): source 세포 2,000개를 유전자 순서를 섞고 비공통 유전자 3개를 붙여 외부 경로로 보내면 source HVG 값과 최대 4.8e-7 차이(float32); 같은 세포를 v1 경로로 보내면 최대 0.497 차이(버그 재현); 공통 non-HVG 유전자에 +100 count → 정정 경로의 0 아닌 HVG 값 전부 감소, v1 경로 불변; zero-library 세포 → 0·유한; 중복 symbol 제외 확인.
- **표현 재구성 T1** (`transform_replay.tsv`): 4개 ALL 표현 모두 shared 목록·HVG 순서·fitmask·cell ID 정확히 일치, source Z와 구 방식 외부 Z의 최대 절대오차 **0**(비트 일치), EVR 동일 → `compatible` 경로(저장된 v1 source Z로 head 적합, Replay 입력 = 저장된 v1 Z_ext).
- **예측 replay T2** (task×model 84개): 81/84 통과. 실패 3개 = sparse_B s2 A11(argmax 일치 0.99892), sparse_Fib s1 A01·A11(0.9245·0.9365). 모두 donor-supported greedy 배분(A01/A11 공유)이 재실행에서 다른 split을 고른 경우다(E1/E2 절의 재현성 관측과 같은 현상). 사전 정책대로 이 2개 task는 같은 3-group source OOF를 새로 적합(`a4_external_norm_v2_oof`)해 새 T·τ를 Replay와 Corrected에 같이 적용했다. 새 T는 14개 중 14개가 v1과 같았고 τ는 9/14 동일, 나머지 5개는 최대 0.0011 차이(sparse_Fib s1 A11 τ_fine)(`calibration_provenance.tsv`). 나머지 10 task는 v1 OOF calibration 재사용.
- **실험 입력/정답**: GSE151530 HCC 25 donor·50,023세포 전체 추론, known 분모 27,212세포(대응 5 label), Malignant 15,723·unclassified 7,088은 출력 분포로만. donor별 class 양성 ≥10 규칙 유지.
- **평가 단위** (`evaluation_unit_counts.tsv`): Original·Replay·Corrected 각각 2,100행, key 중복 0, primary 유효 1,428(head당 nat 75·B 45·Endo 48·Fib 36), seed 평균 donor 700행, 모델 28행, unmapped 168/56행, 누락 key 0. Original 재채점은 v1 `a4_per_donor_seed.tsv`와 최대 5.0e-7 차이(출력 반올림), NA 패턴 동일.
- **입력 변화** (`input_change_summary.tsv`): 외부 세포의 HVG/공통 library 비율 중앙값 0.815(사분위 0.766–0.850; source 0.824) → v1은 외부 값을 평균 약 1/0.81배 크게 만들었다. 정정 후 외부 PCA 좌표 이동 노름 중앙값 0.98(full), 외부 Z 노름 평균 7.65→7.08(source 6.97)로 source 척도에 가까워졌다.

**결과 — 주 대비: 선정 모델 LR_donor, Corrected − Replay (seed 평균 → donor paired bootstrap 10,000)**

| 조건 | Replay | Corrected | 차이 [95% CI] | 동시 98.75% CI | 양/음 donor |
|---|---|---|---|---|---|
| nat (mapped donor-macro AUPRC) | 0.9978 | 0.9972 | −0.0006 [−0.0011, −0.0002] | [−0.0012, −0.0001] | 2/14 (25) |
| sparse_B (target AUPRC) | 0.9256 | 0.9246 | −0.0010 [−0.0025, +0.0004] | [−0.0030, +0.0007] | 5/9 (15) |
| sparse_Endo | 0.9252 | 0.9262 | +0.0009 [−0.0042, +0.0081] | [−0.0051, +0.0102] | 6/10 (16) |
| sparse_Fib | 0.8799 | 0.8602 | **−0.0197 [−0.0328, −0.0084]** | [−0.0368, −0.0060] | 0/12 (12) |

Replay − Original은 LR_donor 4조건 모두 정확히 0이므로 Corrected − Original = Corrected − Replay다.

- 관측: 정정 후 LR_donor 점수는 nat에서 확립됐으나 실용 크기(0.01) 미만으로 낮아졌고, sparse_B·Endo는 차이 미확립, **sparse_Fib는 −0.020으로 낮아졌다(12 donor 모두 음; 실용 크기 이상)**.
- 보류 출력(LR_donor, 같은 calibration): fine coverage nat −0.076, B −0.033, Endo −0.078, Fib −0.002, abstain은 같은 만큼 증가(모든 donor). output error는 nat +0.002 [−0.000, +0.006], B +0.002 [+0.000, +0.004], Endo −0.001 [−0.005, +0.003], Fib 0. AURC는 4조건 모두 +0.006~+0.008(CI가 0 제외). known 세포 중 Hepatocyte 오예측 비율 +0.002~+0.015 증가. 확률 수준: raw 최대 확률 평균 0.90→0.88, known 세포 label 변경률 1.3–1.9%. 같은 calibration에서 coverage가 바뀌므로 fine error는 다른 세포집합에 대한 값이다.
- 다른 head (보조, Corrected − Replay primary): prototype head는 sparse_Endo(+0.020~+0.072)·sparse_Fib(+0.021~+0.040)에서 올랐고(CI가 0 제외), nat A00·A01과 sparse_B P1_donor(−0.016)는 내려갔다. GMM_K는 B·Fib에서 상승. LR_donor만 sparse_Fib에서 내려갔다. 전체 28행은 `paired_impact.tsv`(지표 16개 × 비교 3종).
- Replay − Original(재구성 drift): LR_donor·GMM_K 0; prototype head는 대부분 |평균| < 1e-5, 예외는 T2 실패 task가 섞인 sparse_B A11 +0.0001 [+0.00003, +0.0002], sparse_Fib A01 +0.0060 [+0.0030, +0.0094], A11 +0.0007 [−0.0008, +0.0021].
- Unmapped(LR_donor nat): malignant 세포 fine 비율 0.253→0.163, Hepatocyte 예측 0.913→0.886; unclassified fine 0.268→0.211(`a4_unmapped_cells.tsv`).

**의미**
- 정정 패키지는 완료다: 정규화 계약 일치(구현 검증 10/10), 변환 비트 재현, model/calibration provenance 기록, 2,100 key 완전, drift(Replay−Original)와 정규화 효과(Corrected−Replay) 분리.
- v1 A4의 LR_donor 외부 수치는 잘못된 분모에서 나온 값이었고, 올바른 값은 nat 0.997, sparse B 0.925, Endo 0.926, **Fib 0.860**(v1 0.880)이다. 선정 모델은 LR_donor로 유지한다(외부 label로 아무것도 고르지 않음). prototype head의 외부 점수 상승은 v1 source 선정(A11 미채택)을 소급해서 바꾸지 않는다.
- 가설(미검증): v1 분모가 작아 외부 값이 source보다 크게 표현됐고 그 결과 확률이 과신됐다가 정정 후 낮아졌다(최대 확률·coverage 감소와 일치). LR_donor sparse_Fib 하락의 원인은 측정하지 않았다.

v1 문서에는 원문을 남기고 정정 노트를 추가했다(`reports/r9_rare_state/RESULTS.md` A4 행).


## R9-E1/E2 — prototype 실패 원인 분리 (`reports/r9_rare_state/prototype_diagnostics_v1/`)

**지위**: 이미 반복 개발한 source(GSE149614 바깥 LOPO, 전원 노출)의 후속 원인 진단. 독립 검증 아님. A4 v2 결과는 읽지 않았다. 명세 `RUN_SPEC.md`·`configs/r9/prototype_diagnostics_v1.json`, freeze `5c9340e`(주 대비 6개, 실용 크기 AUPRC 0.01, family별 Bonferroni 동시 98.33% CI, 동등 margin ±0.01(90% CI), 조건부 E2 분산 재적합 규칙).

- **목적**: A10/A11(class당 3성분)이 P1(1성분)·LR/MLP보다 낮은 원인을 (E1) 성분 수 K와 대각 분산 학습, (E2) 강제 fallback과 effective capacity로 나눈다.
- **실험 입력/정답**: sparse B/Endothelial/Fibroblast N64/D4 × 시험 donor 10 × seed 3 = **90 task**, v1 target 제외 fold 표현과 v1과 같은 학습 세포(90/90 design·n_train 일치). 주 지표 = 시험 donor target AUPRC(양성 <10이면 NA). seed를 donor 안에서 평균 → donor별 평가 가능 target 동일 가중 → donor paired bootstrap 10,000회(10 donor). 1,530행(90×17 arm), 누락 0.
- **비교 조건**: E1 P1_current / P3_current / P1_fixedvar / P3_fixedvar / P3_pooledvar; E2 A11_refit / supported_stop / matched_kstar(모두 L_donor,class, 300 step). 기준선 = 같은 key의 v1 저장 예측 LR_class·LR_donor·MLP_donor(+ P1_donor·A10·A11).
- **재현 확인** (`replay_vs_v1.tsv`): P1_current는 v1 P1_donor와 90/90 argmax 완전 일치(최대 |Δp| 8.7e-5). P3_current는 v1 A10과 평균 일치 0.99992(최저 0.9928). A11_refit은 v1 A11과 평균 0.9985(최저 0.970)이고 target AUPRC가 −0.003 [−0.008, −0.000] 낮다: donor-supported greedy가 재실행에서 일부 split을 다르게 골랐다(B·HCC01·s0는 step 10부터 다름; fallback 횟수·시점이 v1과 같은 task 87/90). 사전 명세대로 E2 대비는 같은 실행 안의 A11_refit 기준이다.

**결과 — 주 대비 (target 동일 가중, 10 donor)**

| id | 대비 | 평균 | 95% CI | 동시 98.33% CI | +/− donor | 판정 |
|---|---|---|---|---|---|---|
| E1 a | P3_pooledvar − P1_fixedvar (class 분산 고정 K 효과) | +0.027 | [+0.004, +0.052] | [−0.001, +0.057] | 7/3 | 미확립 |
| E1 b | P3_current − P3_fixedvar (분산 최적화 효과) | +0.013 | [−0.001, +0.028] | [−0.004, +0.031] | 6/4 | 미확립 |
| E1 c | P3_fixedvar − P3_pooledvar (성분별 초기 분산 효과) | −0.014 | [−0.028, −0.003] | [−0.031, −0.000] | 2/8 | **확립·실용 크기 이상(pooled 유리)** |
| E2 d | supported_stop − A11_refit | +0.023 | [+0.010, +0.039] | [+0.009, +0.043] | 10/0 | **확립·실용 크기 이상** |
| E2 e | matched_kstar − A11_refit | +0.025 | [+0.013, +0.041] | [+0.011, +0.045] | 10/0 | **확립·실용 크기 이상** |
| E2 f | supported_stop − matched_kstar | −0.002 | [−0.006, +0.001] | [−0.007, +0.001] | 3/7 | **±0.01 안 동등**(90% CI [−0.005, +0.000]) |

target별(보조): E1 b는 B −0.037 [−0.058, −0.019](고정 분산 유리), Fib +0.066 [+0.033, +0.103](학습 분산 유리), Endo +0.011 [+0.001, +0.021] — target 간 방향이 반대라 평균이 0 근처다. E1 a는 B +0.045 [+0.014, +0.088], Endo·Fib는 CI가 0 포함. E2 d/e는 세 target 모두 양(B +0.050, Endo +0.009, Fib +0.009/+0.016).

보조 대비(95% CI): P3_current − P1_current −0.027 [−0.046, −0.004](현재 방식에서는 K=1이 우세), P3_fixedvar − P1_fixedvar +0.012 [−0.009, +0.032](K와 성분 분산이 섞인 대비, 미확립), P1_current − P1_fixedvar +0.052 [+0.029, +0.078](K=1에서는 분산 학습이 유리), matched_kstar − P3_current +0.027 [+0.006, +0.047].

**강한 기준선 대비 (보조, 95% CI)**: supported_stop − LR_donor −0.004 [−0.021, +0.012], matched_kstar − LR_donor −0.002 [−0.017, +0.013], P1_current − LR_donor −0.002 [−0.020, +0.014] (모두 미확립); 그러나 − MLP_donor는 모든 새 arm에서 −0.060 ~ −0.113(CI가 0 제외), − LR_class는 −0.033 ~ −0.085(CI가 0 제외). 새 arm 중 어느 것도 MLP_donor·LR_class를 넘지 못했다.

**진단** (`capacity_variance.tsv`, `matched_k.tsv`, `fallback_audit.tsv`, `rejection_reasons.tsv`)
- 분산 floor에 붙은 좌표는 모든 arm에서 0. 학습 분산 arm의 |log-variance 변화| 평균 0.36–0.40.
- 3성분일 때 target 성분의 최소 ESS는 B 9.6·Endo 15.1·Fib 14.7(64세포 중), 최대 donor 점유율 0.66–0.79; 학습 CE는 낮아지고(예: B 0.110→0.066) held-out class-balanced CE는 높아져(0.91→1.68) CE gap이 P1 0.80–0.91에서 P3 1.39–1.62로 커졌다.
- Fallback: v1 로그에서 fallback이 생긴 task는 B 26/30, Endo 17/30, Fib 14/30. **target class의 split은 거의 모두 fallback**이었다(v1 B 25/29, Endo 13/14, Fib 8/8). 그 결과 supported_stop의 target 성분 수 K*는 평균 B 1.13·Endo 1.03·Fib 1.00(A11_refit 2.0·1.47·1.27). 첫 fallback 시점의 target 후보는 child donor support <2(B 31/31, Endo 11/16, Fib 11/14)와 held-out gain ≤0으로 거절됐다.
- 조건부 E2 pooled-variance 재적합(조건 c 충족 → 실행, 조건 b 미충족 → fixedvar 재적합 미실행): d +0.006 [−0.003, +0.018], e +0.007 [−0.007, +0.020] 미확립, f −0.001 [−0.010, +0.009](±0.01 동등). pooled 분산은 E2 arm의 절대 점수를 낮췄다(A11_refit_pooledvar − LR_donor −0.037 [−0.058, −0.016]).

**의미 (사전 규칙의 literal 적용)**
- E1: 분산 최적화 효과(b)는 평균으로 미확립이며 target별 방향이 반대다 → "fixedvar만 개선"에 해당하지 않는다. pooledvar가 성분별 고정 분산보다 유리(c, 확립) → plan 규칙상 **희소 성분의 분산 추정/수축이 후속 질문**이다. "K=1이 모든 분산 조건에서 강함"은 성립하지 않는다(현재 방식에서만 K=1 우세; 고정·공유 분산에서는 K=3이 미확립 또는 수치상 높음). 원인은 "성분별 분산 추정과 분산 학습이 64세포 3성분에서 불안정"이라는 가설과 부합하나 확정하지 않는다.
- E2: supported_stop과 matched_kstar가 모두 A11보다 좋고 서로 ±0.01 안에서 동등 → **fallback 제거의 이득은 같은 성분 수의 일반 k-means로도 재현되므로 용량(target 성분 수) 감소의 설명과 일치**한다. 같은 capacity에서 donor-supported partition의 추가 가치는 관측되지 않았다. 이것을 fallback 자체의 인과 효과라고 부르지 않는다.
- 채택: 개선은 P1/A10/A11 대비 원인 분석 결과일 뿐이다. 가장 좋은 새 arm도 LR_donor와 차이 미확립, MLP_donor·LR_class보다는 낮다 → 새 prototype arm 채택 근거 없음. 선정 모델 LR_donor 유지.
- 다음 질문(실행하지 않음): 희소 성분 분산 수축 강도(SHRINK)·class 공유 분산의 별도 사전 고정 대조; greedy 배분의 수치 비결정성(실행 간 split 차이)의 원인 확인. 결과를 보고 grid를 확장하지 않았다.


## §6.4 — 자연 희귀 상태 자료 조건

새 자료 수집 없이 조건만 기록했다: `reports/r9_rare_state/NATURAL_RARE_DATA_CONDITIONS_v2.md` (label 정의·주석 근거·≥3 donor·batch 비혼입·donor 비중복·raw count·무료 공개 접근; 확보 시 N×D 진단과 LR/MLP/P1 기준선 먼저).

## 자원 ledger (`run_ledger_v2.tsv`, `resource_summary_v2.tsv`)

| run | parts | CPU-h | task wall 합 (h) | peak RSS (GiB) |
|---|---|---|---|---|
| c1_transform (4 표현 재구성 + 정정 외부 변환) | 1 | 0.05 | 0.03 | 9.3 |
| a4_external_norm_v2 (84 head 재적합 + replay/corrected 추론) | 12 | 15.37 | 2.00 | 0.81 |
| a4_external_norm_v2_oof (T2 실패 2 task의 source OOF) | 2 | 3.70 | 0.47 | 0.73 |
| prototype_diagnostics_v1 (pilot 3 포함 90 task) | 90 | 120.17 | 15.59 | 0.80 |
| prototype_diagnostics_v1_e2pooledvar (조건부) | 90 | 7.59 | 1.01 | 0.85 |
| 합계 | | **≈146.9** | ≈19.1 | |

평가·집계 job(C2, C1 eval, E1/E2 eval ×2, ledger)은 각 1–2분. GPU 0 GPU-h. 새 디스크 사용 ≈0.6 GB(서버 v2 worktree, v1 자산은 symlink; runs 394 MB, 변환 bundle 191 MB). 동시 job 최대 14(한 차례 16으로 수 초간 초과 — 아래 편차). Pilot(1 donor·seed × 3 target): task당 50–87 CPU-min, wall 6–11 min → 90 task 추정 ≈2 h wall(cap 14) → 전체 실행.

## 편차·사건
- **동시 job cap 일시 초과**: C1 OOF 2 job 제출 시 diag array의 running task가 throttle 축소 전 14개라 r9v2 running이 16이 됐다. 즉시 OOF job을 scancel(시작 직후, 산출물 없음)하고 running ≤12 확인 후 재제출했다.
- **v1 greedy 배분의 비결정성**: 같은 코드·seed·입력으로 donor-supported greedy를 재실행하면 일부 task에서 다른 split을 고른다(A4: 84 중 3 head T2 실패; E2: A11_refit vs v1 A11 최저 argmax 일치 0.970). 원인(BLAS/OpenMP 실행 host 차이 추정)은 확인하지 않았다(가설). 사전 정책대로 C1은 해당 2 task calibration을 새 OOF로, E2는 같은 실행의 A11_refit을 기준으로 썼다.
- E2에서 K*=K(fallback 없음)인 task도 적합을 생략하지 않고 수행해 `kstar_eq_K`로 표시했다(plan은 중복 fit 생략을 허용; 동일성 확인 목적). 
- 계획 제안 경로와 다른 점: 코드는 `scripts/r9/v2/`(제안 `scripts/r9/a2_diagnostics.py`), 트랙 결과는 이 문서. per-target E1/E2 행에도 category가 붙어 있지만 명세상 보조다.

## 미실행 기록
- **E2 frozen-variance(fixedvar) 재적합**: 조건 (b) 동시 CI 상한 <0 & 평균 ≤ −0.01 미충족(+0.013 [−0.004, +0.031]) → 실행 안 함. 해소 조건 없음(사전 규칙상 종료).
- **외부 모델 seed 1/2 (C2 후속)**: 정정의 선행 조건이 아니며 GPU cap 0 → 실행 안 함. 실행하려면 같은 select 표본으로 scPoli 80 task·ProtoCloud 40 task GPU 학습 필요(v1 기준 약 40 GPU-h).
- **ProtoCloud sparse_B/Endo**: v1 미실행, 채우지 않음(같은 이유).
- **자연 희귀 세부 상태 평가(§6.4)**: 조건을 만족하는 다환자 전문가 세부 주석 자료 없음; 조건 문서만 작성.
- **새 prototype 변경의 A3 calibration·A4 적용**: plan §6.3.1상 모델 후보로 남을 때만 확장 — 강한 기준선을 넘지 못해 미실행.

## 열린 결정 (사용자)
1. R9-E1 (c) 결과를 근거로 희소 성분 분산 수축/공유의 사전 고정 대조를 새 버전으로 열지 여부(강한 기준선 대비 이득이 아직 없음을 감안).
2. donor-supported greedy의 실행 간 비결정성을 원인 확인(스레드 고정 재실행)할지 여부 — v1·v2 A11 수치의 재현성에 영향.
3. 외부 모델 seed 1/2 추가(GPU 필요) 여부.

