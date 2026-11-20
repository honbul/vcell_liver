# R9 외부(GSE151530) 출력·보류 지표 errata (v3, 2026-11-20)

근거: private plan 2026-11-20 §5.2·§5.5. 원래 문서·표는 지우거나 고치지 않았다. 이 파일이 정정 기록이다.
대상 표: `reports/r9_rare_state/a4_per_donor_seed.tsv`(v1), `reports/r9_rare_state/a4_model_table.tsv`(v1), `reports/r9_rare_state/a4_external_norm_v2/a4_per_donor_seed_{original,replay,corrected}.tsv`, `a4_model_table_all_versions.tsv`, `paired_impact.tsv`(metric `output_error`), v2 `reports/r9_normalization_prototype_v2/RESULTS.md` 66행.

## 1. 무엇이 틀렸나 (관측)

`scripts/r9/a4_eval.py:108`과 `scripts/r9/v2/c1_a4_eval.py:104`는

```
output_error = n_wrong_output / max(n_output, 1)
```

로 계산했다. known 세포가 있는데 fine·parent 출력이 하나도 없는 행(`n_output = 0`)에서 오류율이 정의되지 않음(NA)이 아니라 **0**으로 기록됐다. 또 행 단위 출력 수(`n_fine, n_parent, n_abstain, n_output, n_wrong_*`)가 표에 없어서 "출력 오류 0"이 출력이 없어서 생긴 0인지 구분할 수 없었다.

| 저장 표 | 행 | 무출력 행 | 그중 output_error=0 기록 | NA 기록 |
|---|---|---|---|---|
| v1 `a4_per_donor_seed.tsv` (Original) | 2,100 | 41 | 41 | 0 |
| v2 `a4_per_donor_seed_original.tsv` | 2,100 | 41 | 41 | 0 |
| v2 `a4_per_donor_seed_replay.tsv` | 2,100 | 41 | 41 | 0 |
| v2 `a4_per_donor_seed_corrected.tsv` | 2,100 | **49** | 49 | 0 |

(`stored_output_error_no_output_rows.tsv`). 무출력 행은 모든 버전에서 **LR_donor × sparse_Fibroblast_N64_D4** 조합에만 있었다. Corrected 기준 75 donor×seed 행 중 49행(seed 0: 25/25 donor, seed 1: 24/25, seed 2: 0/25)이 무출력이며, 이 행들의 known 세포 53,778개(세포×seed)는 전부 보류됐다. 25 donor 모두 출력 있는 seed가 최소 1개 있었다(모든 seed 무출력 donor 0).

`fine_error`는 v1·v2 모두 fine 출력이 없으면 NaN으로 두고 pandas 평균이 NaN을 건너뛰었으므로 값 자체는 이미 NA 규칙과 같았다(차이 ≤ 5e-8). 다만 유효 donor 수가 표시되지 않았다(예: Corrected LR_donor sparse_Fib의 fine_error 0은 7 donor, 3 seed 합계 fine 출력 25세포에 근거).

## 2. 정정 방법

새 행 표(`output_counts_per_donor_seed.tsv`, `_original_replay.tsv`)에 수 필드와 `output_status`를 넣고, `output_error`·`fine_error`·`parent_error`는 분모 0이면 NA. donor-macro 오류율은 donor 안에서 출력이 있는 seed만 평균한 뒤 유효 donor를 동일 가중하고, 유효 donor 수·유효 donor×seed 행 수를 함께 적는다. coverage는 25 donor 전체 동일 가중이다(`RUN_SPEC.md`). 모델·calibration·임계값은 그대로다(T·τ 재계산값이 `calibration_provenance.tsv`와 |차이| ≤ 4.9e-6).

## 3. 영향 (재집계 결과)

- **수치 영향**: 무출력 행이 있던 LR_donor sparse_Fib는 출력이 있는 행의 오류도 모두 0이어서, donor-macro output_error는 정정 전후 모두 0이다(`errata_output_error_old_vs_new.tsv`; 다른 83개 버전×조건×모델 행은 차이 ≤ 5e-8, 출력 반올림). v2 `paired_impact.tsv`의 output_error 대비(28 조건×모델 × 3 대비 = 84행)도 NA 처리 후 값·donor 수 변화 없음(`errata_paired_output_error.tsv`, changed 0행).
- **해석 영향 (정정 필요)**: 아래 기존 서술의 "0"은 출력이 거의 없는 상태에서의 0이다.
  - v2 RESULTS 66행 "output error는 … **Fib 0**"과 v1 `a4_model_table.tsv`·v2 `a4_model_table_all_versions.tsv`의 LR_donor sparse_Fib `output_error 0`: Corrected 기준 출력이 있는 행은 75개 중 **26개**(Original/Replay 34개), donor-macro output coverage **0.111 [0.088, 0.138]**, fine coverage **0.0004 [0.0001, 0.0009]**, 보류 0.889. 출력은 사실상 전부 parent 수준(Stromal/Immune/Parenchymal 묶음)이고, target Fibroblast 세포의 fine 출력은 0개다(`per_class_output_pooled.tsv`). 따라서 "sparse_Fib에서 LR_donor의 출력 오류 0"은 "답을 거의 내지 않았고, 낸 parent 수준 답 8,663개(세포×seed)는 틀린 것이 없었다"로 읽어야 한다.
  - 같은 조건에서 seed별 임계값 차이가 크다: τ_fine 1.01/0.9989/0.9997, τ_parent 0.9974/0.99993/0.9500(seed 0/1/2). seed 2만 parent 출력을 냈다.
- **AUPRC 불변 확인**: 재계산한 class별·allmapped·target·primary AUPRC가 v2 저장 표 3버전과 NA 패턴 동일, 최대 |차이| 5.0e-7(`%.6g` 반올림), 모델 수준 primary 84행 최대 |차이| 5.0e-7(`auprc_unchanged_check.tsv`, `checks_step1.tsv` C5a·C5b). 이번 정정은 AUPRC를 바꾸지 않는다.

## 4. 같은 공식이 남아 있는 곳 (이번 범위 밖, 미정정)

`scripts/r9/a3_eval.py:79`(source LOPO calibration A3)도 같은 `max(n_output, 1)` 공식이다. 저장된 `reports/r9_rare_state/a3_calibration_per_donor.tsv` 400행 중 fine·parent 비율이 모두 0인 행이 13개(sparse_B·sparse_Fib × LR_class·LR_donor)이며 모두 output_error 0으로 기록돼 있다. 외부 출력 유용성(plan §5)의 범위가 아니므로 재집계하지 않았다(`not_run.md`).
