# R9 v3 — P0-R9 출력·보류 지표 정정 (output_utility_v3)

branch `research/r9-output-utility-v3` (base `research/r9-normalization-prototype-v2` @ `0cfd516`). 근거: private plan 2026-11-20 §5, §2, §7, §8.
서버 `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r9_output_utility_v3` (이전 예측은 `vcell_liver_r9_rare_state`·`vcell_liver_r9_norm_proto_v2`로의 읽기 전용 symlink).
Freeze commit(명세·규칙·스크립트, 재집계 전): `e34f466`. 실행 commit `e34f466`, Slurm job 27746.

## 요약 (plan §8 형식)

```text
연구: R9 / 질문: 정규화를 바로잡은 현재 모델(LR_donor)은 기존 신뢰도 정책(source OOF 온도 + 2% source 오류율 임계값) 아래에서 known 외부 세포 몇 개에 답하고, 답한 것 중 몇 개가 틀리는가?
실행: 저장된 GSE151530 예측(7 head × 4 조건 × 3 seed × Original/Replay/Corrected)을 재집계. 무출력 행 output_error 0→NA, 행별 출력 수, donor-macro coverage(25 donor 동일 가중)·조건부 오류(유효 donor·seed 수), 합산 오류율 보조 열. 신규 학습·임계값 변경 없음. 비교 대비 없음(기술 통계).
결과: LR_donor Corrected output coverage [95% donor CI] — nat 0.460 [0.380, 0.546], sparse_B 0.519 [0.443, 0.598](fine 0.182), sparse_Endo 0.405 [0.337, 0.476], sparse_Fib 0.111 [0.088, 0.138](fine 0.0004, 무출력 49/75행). 조건부 output error — nat 0.0025 [0.0002, 0.0061], B 0.0086 [0.0037, 0.0144], Endo 0.016 [0.003, 0.038], Fib 0 (26/75행만 정의). AUPRC 불변(최대 |차이| 5e-7).
판정: 정정 완료. 사용 범위: nat·Endo·Fib = §5.4 1행(출력이 매우 적음 → 전면 자동 분류 채택 보류), sparse_B = 3행(그 조건만 기술; 출력 대부분 parent 수준, coverage CI가 0.50 포함). 조건부 오류가 높다고 확립된 조건 없음. 개발 진단 자료(GSE151530 반복 사용)이며 독립 검증 아님.
다음 행동: 유지(정책·모델 변경 없음, 추가 실험 없음). 운영 요구가 정해지면 다음 독립 평가 전에 최소 coverage·최대 오류·fine/parent 범위를 고정.
사용자 결정 필요: 용도 — 대부분의 세포 자동 답변이 필요한지, 일부 고신뢰(주로 parent 수준 포함) 답변이면 충분한지.
```

## 상세

### 지위와 범위
- GSE151530 HCC 25 donor·50,023세포(known 27,212 / Malignant 15,723 / unclassified 7,088)는 phase2-v2 E00, R9 A4 v1, R9 v2 C1에서 반복 채점된 **개발 진단 자료**다. 이번 재집계로 정책의 독립 외부 검증을 주장하지 않는다.
- 모델·변환·checkpoint·calibration(T, τ_fine, τ_parent)은 바꾸지 않았다. T·τ는 v2와 같은 코드로 저장된 source OOF에서 다시 계산해 `calibration_provenance.tsv`와 일치를 확인했다(최대 |차이| 4.9e-6, 84행).
- 주 버전 Corrected. Original·Replay는 같은 방식으로 보존용 수를 만들었다(`output_counts_per_donor_seed_original_replay.tsv`).

### 무엇을 고쳤나 (`errata.md`)
- v1 `a4_eval.py`·v2 `c1_a4_eval.py`의 `output_error = 오류 / max(출력, 1)` 때문에 무출력 행이 0으로 기록됐다: v1·Original·Replay 표 41행, Corrected 표 49행, 모두 LR_donor × sparse_Fibroblast.
- 수치 영향: 해당 head의 출력 있는 행도 오류가 0이어서 donor-macro output_error와 v2 paired output_error 대비 값은 변하지 않았다(차이 ≤ 5e-8). 해석 영향: 기존 "sparse_Fib LR_donor 출력 오류 0"은 출력 coverage 0.111·fine 0.0004에서의 0이며, 75 donor×seed 행 중 26행에서만 정의된다.
- 같은 공식이 source A3(`a3_eval.py`)에도 있다(무출력 13/400행). 범위 밖이라 재집계하지 않았다(`not_run.md`).

### 결과표
- 주 결과: `usability_decision.md` §1 표(LR_donor Corrected, 4 조건). 전체 28 head×조건 × 3 버전은 `coverage_error_summary.tsv`(coverage·오류 CI, 유효 donor/행 수, Σ수, 합산 보조 오류율, 대응 없는 세포 비율, seed별 T·τ).
- 원표: `output_counts_per_donor_seed.tsv`(Corrected 2,100행), `output_counts_per_donor_seed_original_replay.tsv`(4,200행). 25 donor 전체 coverage: `coverage_per_donor.tsv`. 무출력: `no_output_summary.tsv`. 대상 class별(보조, 세포 합산): `per_class_output_pooled.tsv`. 판정 대응: `decision_mapping.tsv`.
- 보조 관측(다른 head, Corrected): GMM_K의 output coverage는 nat 0.781, B 0.675, Endo 0.772, Fib 0.765로 LR_donor보다 높고 조건부 오류는 0.0004–0.0075로 관측됐다. A00·A01·A10·A11·P1_donor는 sparse_Endothelial에서 조건부 오류 0.028–0.045(점추정)가 관측됐다. **외부 결과로 모델을 바꾸지 않는다**(plan §5.3); 선정 모델은 LR_donor 그대로다.

### 검사 (`checks_step1.tsv`, `checks_step2.tsv`; 모두 통과)
| 검사 | 결과 |
|---|---|
| C1 fine+parent+abstain = n_known (6,300행), 비율 합 = 1 | 통과(최대 |합−1| 2.2e-16) |
| C2 수 항등식, 오류 ≤ 출력 | 통과 |
| C3 (버전,조건,seed,모델)별 Σknown 27,212·Malignant 15,723·unclassified 7,088; 버전별 2,100행, 중복 0 | 통과(252 group) |
| C4 분모 0 ⇔ NA | 통과(무출력 131행 = 49+41+41, no_known 0행) |
| C5 AUPRC(class별·allmapped·target·primary) 저장 v2 표와 동일, 모델 수준 primary 동일 | 통과(최대 |차이| 5.0e-7, NA 패턴 동일) |
| C6 저장 fine/parent/abstain 비율·fine_error·출력 있는 행 output_error 동일 | 통과(5.0e-7) |
| C7 T·τ = calibration_provenance | 통과(4.9e-6) |
| S1–S5 요약은 원표만 읽어 재생성, 비율 재계산 일치, donor-macro 합 = 1, errata 행 매칭 | 통과 |

입력 버전: `input_hashes.tsv`(외부 obs h5ad 1개, v1 예측 12개, v2 예측 12개 + run.json 12개, v2 OOF 2개의 sha256).

### 자원 (`run_ledger.tsv`)
| run | job | wall | CPU 시간 | peak RSS |
|---|---|---|---|---|
| pilot (1 task) | 27743 | 0:34 | 20 s | 0.32 GiB |
| main (step 1+2) | 27746 | 3:25 | 218 s | 0.36 GiB |

할당 4 CPU 기준 약 0.27 CPU-h(실사용 CPU 시간 약 0.07 h). GPU 0. 동시 v3r9 job 최대 1. 새 대용량 자산 없음(symlink만).

### 편차·사건
- `decision_mapping.tsv`의 `error_vs_0.02_undetermined` 표기는 RUN_SPEC 문구("B·C 모두 아니면 미확립")대로 붙었으나, config의 괄호 설명("CI contains 0.02")과 어긋나는 경우가 있다: nat(CI 상한 0.006)·sparse_Fib(0)는 CI가 0.02 아래이며 coverage 때문에 3행이 아닌 것이다. 결정은 바뀌지 않으며 표 문구는 결과 후 수정하지 않았다(`usability_decision.md` §2).
- 서버 git에 commit identity가 없어 결과 TSV는 서버에서 rsync로 받아 로컬에서 commit했다(내용 동일).

### 미실행 (`not_run.md`)
비결정성 replay(불필요), A3 source 표 재집계(범위 밖), LR_class·MLP 외부 비교(예측 없음, 별도 사이클), 새 임계값·합격선(금지), 독립 외부 검증(자료 없음).
