# R8 v3 — E2-B 미관측 약물 선택 분리 정정 결과

**작성일:** 2026-11-20 · 브랜치 `research/r8-unseen-drug-selection-v3` (기준 `research/r8-observation-transfer-v2` @ `b419b3e`) · **freeze `f0589c6`** · 근거: 비공개 기획서 2026-11-20 §4, §7, §8.

## 요약 (6줄)

**P0-R8**
```text
연구: R8 P0 / 질문: E2-B에서 outer query 약물을 적합뿐 아니라 상수 선택에서도 완전히 뺄 수 있는가?
실행: outer(family 3 × v2 drug fold 3) 안에 outer 허용 약물만으로 inner drug fold 3개를 새로 만들고, alias closure를 적용해 f0 표·PCA·kernel scale·ridge·선택을 모두 허용 범위로 제한; ID 교집합 검사 351개.
결과: v3 검사 위반 0/351; v2 inner 선택은 block마다 outer query 약물 416–775개를 정답으로 썼음(누수 확인); outer episode는 v2와 동일; 9/9 block 적격(16–22 family).
판정: 완료 (P1 시작 조건 충족).
다음 행동: P1-R8 실행 (완료).
사용자 결정 필요: 없음.
```

**P1-R8**
```text
연구: R8 P1 / 질문: 같은 약물 반응을 적합·선택 어디에서도 보지 않아도 화학 경로(ridge_C1chem)가 fallback(fb_C0)보다 약물별 차이를 더 잘 전달하는가?
실행: 분리 정정 계약에서 κ_f·κ_b·ridge α·λ·η를 block별로 재선택, K=16, 4 arm, outer seed 1–3, 71 family·6,723 약물·38,302 query; 주 대비 centred RMSE, two-way family×drug paired bootstrap.
결과: fb_C0 0.54352 → ridge_C1chem 0.54086, 차이 −0.00266 [−0.00394, −0.00142] (상대 −0.489%, fold 3/3); 전체 ΔRMSE −0.00468 [−0.00726, −0.00240].
판정: 채택 보류 — 작은 개선 (방향 확인, 상대 감소가 사전 기준 0.5%에 미달).
다음 행동: 종료 (이번 미관측 약물 경로 모델 탐색 종료, C1은 기존 용도에서 유지).
사용자 결정 필요: 없음 (미관측 약물 기능 자체가 불필요하면 해당 주장 철회 가능 — 기획서 §0 기본값 1).
```

**노출 지위.** 모든 family·약물은 R8 / v1 / v2에서 이미 분석된 source 자료다. 이번 결과는 **정정된 개발 검증**이며, 새 독립·blind 결과가 아니다.

## 1. P0-R8 — 분할·선택 경로 수정

**문제(v2).** `e2b.py sru`의 inner 선택이 `held_episodes(..., held, "val", ...)`로 outer와 같은 fold j 약물을 inner-val family의 query로 채점했고, κ_f는 full-reference P0 선택값을 썼다(`old_vs_corrected.md`).

**수정(`src/r8_unseen_drug_v3/core.py`).**
- outer: v2 분할 그대로(family = cells.tsv cv_fold, drug = crc32('r8v2_drug|'+group) % 3). query 집합은 v2와 동일.
- alias 예외 목록(`configs/r8_unseen_drug_v3/alias_exceptions.tsv`): canonical group이 다르지만 largest-fragment SMILES 또는 InChIKey 첫 블록이 같은 6 cluster(12 group). query 약물의 alias는 해당 block의 모든 적합·support·선택에서 제외(block당 +2~5 group).
- inner: outer 허용 약물 A = closure(fold j)의 여집합을 crc32('r8v3_inner_drug|'+group) % 3으로 3분할. inner fold i의 query = A ∩ fold i(inner-val cell, seed 1, cap 200), 적합 제외 = closure(fold j) ∪ closure(A ∩ fold i).
- 모든 context(outer block, inner fold 각각)의 f0 표·f0 PCA·kernel scale·ridge 사상은 그 context의 train family와 허용 group으로만 적합. 대조군 PCA는 label-free x0만 사용. covariance nuisance는 만들지 않음(사용하는 arm 없음).

**검사 결과.**

| 검사 (`p0/overlap_checks.tsv`) | 위반 |
|---|---|
| outer query (alias closure) ∩ outer f0 표·PCA·ridge / kernel-scale episode / outer support | 0 |
| outer query (alias closure) ∩ inner query / inner support / inner 적합 | 0 |
| inner query (closure) ∩ inner f0·PCA·ridge / scale / support | 0 |
| episode 내 support·query group 교집합 | 0 |
| outer-val ∩ outer-train family, inner family ⊄ outer-train, 채점 family ∩ 적합·선택 family | 0 |
| (참고) v2 inner 선택 query ∩ outer query group | 416–775 / block (누수 기록) |

- outer episode: 9 block 모두 (cell, seed)별 query 수·support 수·적격 여부가 v2 `eligibility.tsv`와 일치.
- 적격(`p0/eligibility_blocks.tsv`): 9/9 block, 적격 family 16–22 (최소 10), inner 선택 episode 21–38 (8–13 family).
- 적합 범위(`p0/fit_scope.tsv`): context별 family·group 수와 목록 hash, outer query group 포함 수(모두 0).
- 분할 파일: `p0/outer_splits.json`, `p0/inner_splits.json`, `p0/drug_folds.tsv.gz`.

## 2. P1-R8 — 분리 정정 후 재평가

**입력/정답.** outer0–2 × drug fold 0–2, outer-val family의 일반 episode(seed 1–3), K = 16(support 앞 16개), query = fold j 약물. 478 episode, 71 family, 6,723 약물 group, 38,302 query. query 처리 후 반응은 채점에만 사용.

**선택된 상수**(`p1/selected_constants.json`, `p1/constants_v2_vs_v3.tsv`): κ_f는 9 block 모두 32(grid 상한), ridge α 1000(6 block) / 10000(outer1 3 block), κ_b 16–64, λ 0.01–3, η 0.25–0.75. 선택 grid의 exact tie 0건(최소 1·2위 차 2.8×10⁻⁶).

**arm 평균 (K = 16, family 평균).**

| arm | centred RMSE | ΔRMSE |
|---|---:|---:|
| fb_C0 | 0.543523 | 0.565477 |
| fb_C1chem | 0.542183 | 0.564006 |
| ridge_C0 | 0.542192 | 0.562175 |
| ridge_C1chem | 0.540864 | 0.560795 |

**주 대비와 보조 대비** (`p1/primary_contrast.tsv`, `p1/secondary_contrasts.tsv`; CI는 two-way family×drug, 괄호 안 family-only는 참고).

| 대비 | 지표 | 차이 | two-way 95% CI | 상대 | fold (outer0/1/2) |
|---|---|---:|---|---:|---|
| **ridge_C1chem − fb_C0 (주)** | centred | **−0.00266** | **[−0.00394, −0.00142]** ([−0.00384, −0.00169]) | **−0.489%** | −0.00382 / −0.00079 / −0.00336 (3/3) |
| ridge_C1chem − fb_C0 | ΔRMSE | −0.00468 | [−0.00726, −0.00240] | −0.83% | 3/3 |
| ridge_C0 − fb_C0 (anchor) | centred | −0.00133 | [−0.00215, −0.00059] | −0.24% | 3/3 |
| ridge_C1chem − ridge_C0 (잔차) | centred | −0.00133 | [−0.00237, −0.00043] | −0.24% | 3/3 |
| fb_C1chem − fb_C0 (기술) | centred | −0.00134 | [−0.00244, −0.00041] | −0.25% | 3/3 |

**판정 적용.** CI 상한 < 0, fold 3/3, 평가 fold 3/3, 전체 ΔRMSE 개선 — 그러나 상대 감소 0.489% < 0.5% → **작은 개선, 채택 보류**. 기준은 동결된 값을 그대로 적용했다.

**재집계 검증.** 저장된 예측(`p1/prediction_assets.tsv`: 서버 경로·sha256, block별 `preds.npz` 합계 약 750 MB)에서 Gram 경로로 계산한 episode centred/ΔRMSE가 실행 시 기록한 값과 최대 5.0×10⁻⁹ 차이, family 평균은 2.0×10⁻¹⁰ 차이(`p1/consistency.json`). 집계 순서: episode → cell(seed × drug fold) → family → 전체.

**관찰 / 가설 / 미검증.**
- 관찰: 선택 누수를 없애도 centred 개선 방향과 크기(−0.0027)는 v2(−0.0024)와 비슷했다. anchor와 잔차 기여가 거의 같은 크기(각 −0.0013)로 더해진다.
- 관찰: outer1 fold의 개선(−0.0008)이 다른 두 fold(−0.0034~−0.0038)보다 작다.
- 관찰: κ_f가 모든 block에서 grid 상한(32)으로 선택되었다. 더 큰 κ_f가 inner에서 더 나았을 가능성은 검증하지 않았다(grid 동결).
- 가설(미검증): outer1의 작은 효과가 family 구성 차이 때문인지 등 원인은 측정하지 않았다.
- 미검증: 외부 독립 자료, 다른 K, matched plate 조건, 다른 화학 표현.

## 3. 이전 결과와의 관계

v2 E2-B sru 결과는 **"예측 시 참조 부재 실험"**으로 그대로 보존한다. 비교와 바뀐 주장은 `old_vs_corrected.md`, v2 보고서용 정정 안내는 `reports/r8_transfer_v2/ERRATA_E2B_v3.md`.

## 4. 자원·편차

| 항목 | 값 |
|---|---|
| Slurm job | 19개 (`run_ledger.tsv`, `p1/slurm_jobs_raw.tsv`): 완료 17, 실패 2 (27769 env 경로 — 계산 전; 27771 report 변수 이름 버그 — 주 표 작성 후 consistency 단계에서 중단, `report_v2`로 대체) |
| CPU | 실제 user+sys 약 0.83 CPU-시간, 예약 core-시간 약 1.4 (4 core × 경과); 첫 job 제출 09:50, 마지막 job 종료 10:02 (약 12분) |
| 메모리 | 최대 7.1 GB/job |
| GPU | 0 |
| 디스크 | 서버 `assets/` 720 MB (예측 저장 포함) |
| 동시 job | 최대 8 (2026-11-20 사용자 요청으로 track 상한 2 → 8; 설계 변경 없이 block별 분할) |

**편차·사건.**
1. P0 검사 첫 실행(`chk_*_v1`)의 v2 참고 행이 unit 번호와 group 번호를 섞어 0으로 기록됨 → 코드 수정(`38d688f`) 후 `chk_*_v2`로 재실행. v3 검사 자체는 두 실행 모두 0.
2. `run_outer2_d2_v1`(job 27769)은 로그인 홈 디렉터리에서 제출되어 env 파일을 찾지 못하고 즉시 종료(계산·출력 없음) → 새 run ID `_v2`로 재실행.
3. `report_v1`(job 27771)은 주 표를 쓴 뒤 consistency 단계에서 변수 이름 충돌로 중단 → 코드만 수정(`36b5998`), `report_v2`로 한 번 다시 계산. 수치는 `report_v2` 기준.
4. 로그인 노드에서 실행한 것: alias 목록(약 10초, drugs.tsv), splits(메타데이터), 소형 TSV 수집.

**미실행.** `NOT_RUN.md`.
