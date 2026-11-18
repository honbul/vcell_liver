# R10 — 구성비와 상태 내부 변화를 분리한 용량 보간 (Sci-Plex3)

**무엇을 예측하나.** 같은 약물·같은 세포주에서 두 용량(endpoint)을 실제로 측정했을 때, 그 사이의 측정하지 않은 용량(query)에서 관측될 세포 집단 분포를 예측한다. 출력은 고정 상태 atlas에서의 상태 구성비와 상태 내부 분포(및 이를 조립한 세포 구름)다.

| episode | 입력 endpoint | 예측 대상 | 비고 |
|---|---|---|---|
| E1000 | 100 nM + 10 µM | 1 µM | 주 평가 |
| E100 | 10 nM + 1 µM | 100 nM | 별도 표 |

**허용 정보.** 시험 약물의 endpoint 두 well 세포(train-only PCA 좌표), endpoint replicate의 vehicle 세포, 요청 용량, 세포주. query well의 RNA·평균·구성비·세포 수·depth는 예측 입력과 설정 선택에 쓰지 않는다(코드의 `Context` 객체에 query가 없다). 이것은 **부분 측정 약물의 미측정 용량 예측**이며, 완전 미관측 약물(zero-shot), 새 세포주, 성인 HCC 예측이 아니다. A549·K562·MCF7은 일반 암 세포주다.

**자료.** GEO GSE139944 / GSM4150378 원본(pData·UMI 행렬·hash 메타데이터)만 사용. 수신 기록·해시는 [`DATA_CONTRACT.md`](DATA_CONTRACT.md)와 `manifests/r10_dose_composition/raw_data_receipt.json`.

**기획서.** 비공개 연구 기획서(Git에 넣지 않음): 448줄, 35,257 bytes, sha256 `aaecd39d95f967d7883a0bcdf18bb029f23da38cd401cda77ba49805f502f884`. 이 보고서는 그 §4–10 설계와 §12 산출물 규칙을 따른다.

**현재 단계 (2026-11-18).** B0–B3 완료, cycle C1(2×2 v1)·C2(same-replicate 재학습)·C3(전역 mixing weight 기준선) 완료. development 규칙으로 **D11 v1을 선택해 freeze(commit `f0ded27`) 후 test 약물에 1회 평가**했다. 추가 cycle은 열지 않았다. 결론과 범위는 [`RESULTS.md`](RESULTS.md) §0과 [`CLAIMS_AND_LIMITATIONS.md`](CLAIMS_AND_LIMITATIONS.md).

**문서.**

| 파일 | 내용 |
|---|---|
| [`DATA_CONTRACT.md`](DATA_CONTRACT.md) | 원본·QC·단위·분할·허용 문맥·지표 |
| [`DECISIONS.md`](DECISIONS.md) | cycle별 관측 → 가설 → 변경 → 예상(실행 전 작성) → 결과 |
| [`RESULTS.md`](RESULTS.md) | 관측 / 추론 / 미검증을 나눈 결과 |
| [`CLAIMS_AND_LIMITATIONS.md`](CLAIMS_AND_LIMITATIONS.md) | 주장 가능한 것과 아닌 것 |
| [`BEGINNER_SUMMARY.md`](BEGINNER_SUMMARY.md) | 쉬운 한국어 요약 |
| [`BASELINES.tsv`](BASELINES.tsv) | 같은 정보 계약의 기준선, 정보·출력 형태·비용(triplet당 초, 채점 포함)·development 성능 |
| `dose_replicate_map.tsv` | 24 h culture well 4,608개(세포 0 well 포함)의 replicate·plate·용량·QC 세포 수 |
| `eligible_triplets.tsv` | 세포주×약물×episode×replicate 방향의 적격성과 제외 이유 |
| `replicate_components.tsv` (+`_pairs.tsv.gz`) | 반복 well·vehicle·용량·약물 쌍의 구성비/상태 내부/전체 분포 차이 |
| `composition_metrics.tsv.gz`, `within_state_metrics.tsv.gz`, `assembly_consistency.tsv` | 조건 단위(role×세포주×약물×episode×방향×방법; 두 방향·seed 평균) 지표. 앞의 둘은 크기 때문에 gzip |
| `summary_by_method.tsv`, `paired_vs_D00.tsv`, `summary_by_line.tsv`, `seed_level_summary.tsv` | 약물 단위 평균·paired bootstrap 구간(D00 및 nearest_low·state_moment·mixture_t 대비) |
| `mixture_failure_map.tsv`, `mixture_failure_classes.tsv`, `endpoint_budget.tsv`, `decomposition_train_conditions.tsv.gz`, `K_selection.tsv`, `synthetic_controls.tsv` | B1·B2 진단 (`synthetic_controls.tsv`의 `case=null`은 문자열; pandas에서는 `keep_default_na=False`) |
| `c2_same_contract_paired_vs_D00.tsv` | C2(same 계약 재학습) paired 표 |
| `eval_inputs.json` | 최종 집계에 들어간 run 파일 목록 |
| `run_ledger.tsv` | Slurm job·git SHA·시간·메모리 |
| `figures/` | §11 그림 |

**재현.** 서버 worktree에서 `sbatch scripts/r10/b0_prepare.sbatch` → `scripts/r10/atlas.sbatch` → `scripts/r10/py.sbatch r10_dose_composition.<module> ...`(인자는 `run_ledger.tsv`의 `cmd` 열). 모든 대용량 산출물은 `assets/`(Git 제외) 아래이며 sha256 색인은 `manifests/r10_dose_composition/server_artifacts.tsv`.

**코드 재사용.** R3(`research/hcc-anchored-distribution-r3`, `src/anchored_distribution_r3/metrics.py`, commit 0ace11c, sha256 dd6475e5…)의 energy distance 추정량과 약물 블록 bootstrap을 복사·일반화했다(`src/r10_dose_composition/core.py` 머리말). 브랜치 병합은 하지 않았다.
