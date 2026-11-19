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

__C1__

__E12__

## §6.4 — 자연 희귀 상태 자료 조건

새 자료 수집 없이 조건만 기록했다: `reports/r9_rare_state/NATURAL_RARE_DATA_CONDITIONS_v2.md` (label 정의·주석 근거·≥3 donor·batch 비혼입·donor 비중복·raw count·무료 공개 접근; 확보 시 N×D 진단과 LR/MLP/P1 기준선 먼저).

__LEDGER__
