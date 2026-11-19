# R9 v2 — 외부 정규화 정정 · matched 외부 모델 대비 · prototype 실패 원인 분리

branch `research/r9-normalization-prototype-v2` (base R9 tip `1d863eb`). 근거: private plan 2026-11-19 §6, §2, §8, §9.
서버 `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r9_norm_proto_v2` (v1 자산은 `vcell_liver_r9_rare_state`에서 읽기 전용 symlink).
Freeze commit(R9-C1·E1/E2 실행 명세, 외부/source 점수 산출 전): `5c9340e`.

__SUMMARY__

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


__E12__

## §6.4 — 자연 희귀 상태 자료 조건

새 자료 수집 없이 조건만 기록했다: `reports/r9_rare_state/NATURAL_RARE_DATA_CONDITIONS_v2.md` (label 정의·주석 근거·≥3 donor·batch 비혼입·donor 비중복·raw count·무료 공개 접근; 확보 시 N×D 진단과 LR/MLP/P1 기준선 먼저).

__LEDGER__
