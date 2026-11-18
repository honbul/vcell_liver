# R8 후속 분석 v1 — 버전과 자료 노출 기록 (VERSION_AND_EXPOSURE)

**작성일:** 2026-11-18 (한국시간) · 코드 점검용 개발 실행(V0 재현 `p1_dev1`, source fold 점검 `*_dev1`)만 한 시점, 분기 결정·동결 **전**, 확인 family·HCC 값을 읽기 **전**에 작성했다. 이후 변경은 §5 변경 기록에만 추가한다.

## 1. 기준 버전

| 항목 | 값 |
|---|---|
| 후속 브랜치 | `research/r8-followup-observation-transfer-v1` |
| 기준(원 R8 최종) 커밋 | `5530acb9f1536dfcbd5c5427728ad7d748ce82d8` |
| 원 R8 동결 커밋 / 동결 코드 | `5a3b80988c97b4d478bba6c3fb0e783df8bbc686` / `43500fcdd781c27cfc3bb148ab1c27bbdd857d05` |
| 원 R8 HCC 평가 | Slurm job 25686 (`hcc_v1`, 한 번) |
| 후속 기획서 (비공개, Git에 넣지 않음) | 817줄, 70,998 bytes, SHA-256 `b89ea16e9b77d46cd2fce21f0e32e87f0557a850fa984ffb5f8c640a0361b56d` |
| 기획서가 언급한 로컬 `audit_recalculation.json` | 로컬에서 찾지 못했다. 내용을 추정하지 않고 필요한 값(V1 민감도 등)은 모두 다시 계산했다 |

원 R8의 코드(`src/replicate_calibrated_r8/`), 보고서, manifest, 서버 `assets/`(store·checkpoint·예측)는 **읽기 전용**이다. 후속 코드는 `src/r8_followup_v1/`, 결과는 `manifests/r8_followup_v1/`·`reports/r8_followup_v1/`, 서버 산출물은 후속 worktree의 `assets/`에만 쓴다. `diag.py`의 공분산 식 수정과 fold별 Context 경로는 원 파일을 고치지 않고 후속 namespace의 복사·확장 코드(`covrepair.py`, `engine.py`)로 구현한다. 따라서 원 V0 결과는 원 코드로 그대로 재현된다.

## 2. 자료·분할 hash

| 파일 | SHA-256 |
|---|---|
| store `cells.tsv` (3-fold family manifest `cv_fold` 포함) | `bf8ccd3c12f48b45d68b4ca8e3f5a0e981f75020087fc7e1a480f0a504e40fac` |
| store의 나머지 파일 | `manifests/replicate_calibrated_r8/store_manifest.json`의 hash를 매 job 시작 시 대조 (`partitions.Bank`) |
| 원 feature basis (`feature_basis.npz`, 전체 source-train 적합) | `f20e32ff211aeb7e4283ea038a4ab1847451cd8a0e66746f56c1d7ff14bbd1e1` |
| 원 plate 쌍 표 (`runs/diag/diag_v1/d1_plate_pairs.tsv`) | `9af18e1eccf22995e7fbf825797004c92a39396bb849b0d3de78d107e4a47035` |
| `reference_families.tsv` | `721eeab24634f9543e951026f062c7406c3dd0a323ffa9903d3c414eb7c1a1ff` |
| `anchor_selection.json` (원 상수, fixed-hyper 행에서만 사용) | `4abc119add1e2f208e59a82067edc99d9e34deb1f9b09a79da7f86a8b947c034` |
| `covariance_model.json` (원 V0) | `9629ebe901d5d3b0ebfc9b6264c25971bcf9b8e4c56dbb891c81dd37c32c2158` |
| `freeze_manifest.json` (원 HCC 분할·support 목록) | `e7325ae1c5e4f6cbe59fc6f0bdb732dfdad51063018ee664982c8a2c48062f26` |
| 원 `protocol.yaml` | `25c79d047f0d1796fb66b39e2d3a1c77773ad14c580814d27f1f04abd811f188` |

CMap LINCS 2020 Level 3 배열은 연구 B·R2의 처리 자산(읽기 전용)을 그대로 쓴다. 공유 pubdata는 읽기 전용이며 다시 받지 않는다.

## 3. 이미 본 결과 (이 후속의 설계 동기에 영향을 준 것)

| 자료 | 이미 본 것 | 후속에서의 취급 |
|---|---|---|
| HCC 세 계열 (HUH7, JHH5, JHH7) | 연구 B, R2, R8에서 K=8/16/32 점수를 모두 봤다. 기획서 §2.1의 표가 그 값이다 | **이미 열람된 패널.** 동결 커밋 이후 동결 후보와 기준선에 한 번만 적용한다. "기존 벤치마크 후속 평가"라고 부른다. target 검증 라벨 0개 |
| source 확인 9 family | R8 `source_eval_*`, `closed_form_summary.tsv`(confirmation 행)에서 봤다 | 동결 후에만 동결 후보와 기준선에 한 번 적용한다 |
| 확인 5 family의 같은/다른 plate 결과 | R8 `matched_plate_transfer.tsv`(기획서 §2.2의 표) | 동결 후에만 적용한다. 분기 선택에는 쓰지 않는다 |
| source-train family의 같은/다른 plate 결과 (41 family) | R8 `matched_plate_transfer.tsv`의 `train_closed_form` 행 | V0 재현에만 쓴다. 후속의 E1/E2 판단은 fold별로 다시 적합한 out-of-fold 값으로 한다 |
| source-train dev 점수 | R8 `closed_form_summary.tsv`(dev 행) | V0 재현에만 쓴다 |
| 공분산 공식 불일치 | 기획서가 공개 요약으로 계산한 민감도 0.0360750 | **하드코딩하지 않는다.** 원 코드·자료로 다시 계산해 비교만 한다 |

즉 이 후속의 질문(공분산 정합성, 같은/다른 plate의 차이, query 관측 환경)은 **이미 본 source 확인·HCC 결과에서 나왔다.** 분기 선택은 source-train family의 out-of-fold 결과만으로 하고, 이미 본 패널의 점수로 상수·분기·gate를 고르지 않는다.

## 4. 정보 경계 (fold별)

- 보류 단위는 **세포주 family**다(R6/R7의 약물 보류 규칙이 아니다).
- outer: 기존 3-fold family manifest. fold k의 family가 평가 대상이고, f0/f*, PCA 축, 공분산(V0–V3의 v_e·v_s·v_r), kernel block scale은 나머지 두 fold의 family로만 적합한다.
- inner: outer-train family 안에서 `crc32("r8f_inner|" + family) % 3 == 0`인 family를 inner-validation으로 고정한다. κ_f, κ_b, λ, η, J1/J2 상수는 inner-train으로 적합한 nuisance 위에서 inner-validation 점수로만 고른다. 선택 후 outer-train 전체로 nuisance를 다시 적합한다.
- 원 코드의 전역 `anchor_selection.json`·`covariance_model.json` 경로는 엄격한 out-of-fold 평가에서 쓰지 않는다. 원 상수는 V0 재현과 "fixed-hyper" 행에만 쓰며, 그 행은 원 상수가 모든 source-train family(평가 fold 포함)로 골라졌으므로 **source 개발 분석**으로 표시한다.
- 동결 후보의 최종 상수는 같은 inner 규칙을 115개 source-train family 전체에 적용해(`final_inner`) 고르고, nuisance는 source-train 전체로 적합한다(원 R8과 같은 정보 범위).

## 5. 변경 기록

| 날짜 | 변경 | 사유 | 그때까지 본 것 |
|---|---|---|---|
| 2026-11-18 | 문서 작성 | — | 원 R8 결과(위 §3)만 |
| 2026-11-18 | 코드 점검 개발 실행에서 J1/J2 OOF 값이 계산·열람됨 | J 모형이 공통 method 집합에 연결되어 있었음. 공식 P1/P2 실행(`4c2da64`)에서는 제외 | source-train OOF만 (확인 family·HCC 미열람). `BRANCH_DECISION.md` §7 |
| 2026-11-18 | 분기 결정 커밋 | §8 + §10 선택 | source-train OOF P1/P2 결과 |
