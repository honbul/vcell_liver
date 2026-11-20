# R3 — Tahoe-100M SNU-423 화학적 LOCO 소규모 실행 (pseudobulk 기준선)

**작성:** 2026-11-10 · 계획 [FEEDBACK_PLAN_20261110.md](../../docs/FEEDBACK_PLAN_20261110.md) §4-2, §5 R3, 실행 권고 "R3의 SNU-423 소규모 추출" · 선행 자료 [F4_LIVER_DATA_CARD.md](F4_LIVER_DATA_CARD.md), [F3_CROSS_CELL_LINE.md](F3_CROSS_CELL_LINE.md) · 커밋하지 않음.
**성격:** 아래 수치는 모두 이번 작업에서 G1 Slurm으로 계산한 **실측**이다. 모델 학습은 단순 기준선(평균·kNN·ridge·kernel ridge)까지만 했다. 사전학습 모델(State, PIE 등)은 **실행하지 않았다**. 상세 수치는 [`R3/`](R3/)의 TSV/JSON에 있다.

## 0. 요약

| 항목 | 결과 |
|---|---|
| A1 pseudobulk | 로컬 Tahoe-100M 샤드 3,388개를 **한 번** 읽어 50개 세포주 × 1,344 well = 67,200개 (well, 세포주) 그룹의 원 count 합과 세포별 CP10k 합을 만들었다. 세포 **95,624,334개**를 모두 집계했고 obs_metadata의 `pass_filter=full` 세포 수와 **그룹 단위로 0건 불일치**다. `minimal` 세포는 샤드에 0개 들어 있다. SNU-423 1,344 well의 합은 F4의 독립 추출(h5ad 기반)과 **완전히 일치**한다(최대 차이 0 count). 18분 05초(c3, 48 CPU 중 worker 42개, worker 합계 6.2 CPU-h), 산출물 32 GB. |
| F4 전제 정정 | "샤드 1개 = plate 1개"는 정확하지 않다. **13개 샤드가 plate 두 개에 걸쳐 있다.** 행을 plate로 걸러 처리했다. |
| 핵심 발견: well 공유 | 같은 well의 학습 세포주 평균은 SNU-423 정답과 섭동 중심 r **0.77**을 보인다. 그런데 이 수치는 반응이 DMSO 잡음과 구별되지 않는 조건(0.75)이나 **약물이 없는 DMSO well(음성 대조, r 중앙값 0.76, 28 well 모두 0.64–0.88)**에서도 거의 같다. 즉 이 신호의 대부분은 MOSAIC 공동배양 well이 공유하는 **기술적 well 효과**다. Tahoe에서 "관측된 drug×dose"는 975/1,137 조건에서 곧 "같은 well"이므로, 조건 (a)를 단순 LOCO로 평가하면 이 효과가 성능으로 잡힌다. |
| (a) 미관측 세포주·관측 약물 | **well 공유(W, 오라클 유사):** kNN(20개 세포주) 같은 drug×dose 평균의 RMSE 0.0753, no-change 0.1342(ΔRMSE −0.059), 섭동 중심 r 0.79, retrieval rank 0.000. **well 비공유(plate 분리, aP):** 최선 `hier_offplate_calibrated` RMSE 0.1188, no-change 0.1356(−0.017 [−0.019, −0.015]). `same_drug_offplate`의 섭동 중심 r은 0.38, plate 중심 r은 0.46이다. 검출 가능 조건에서는 −0.023, 섭동 중심 r 0.41이다. |
| (b) 미관측 세포주 + 미관측 약물 | 최선 비공유 기준선은 `dose_mean`(학습 세포주·학습 약물의 같은 용량 평균)으로 RMSE 0.1185, no-change 0.1292(−0.011)다. Morgan 지문 kernel ridge(0.1187)는 이를 넘지 못한다(+0.0002 [−0.0005, +0.0010]). 섭동 중심 r ≤ 0.16, plate 중심 r ≤ 0.12다. |
| (c) SNU-423 내부 미관측 약물 | `dose_mean_snu` RMSE 0.1148, 지문 ridge 0.1151이다. (b)보다 모든 층에서 조금 낫다. 교차 − 내부 RMSE는 +0.0037 [+0.0025, +0.0048]이고, 섭동 중심 r은 −0.027이다. 두 경우 모두 새 약물에 대한 약물 특이적 예측은 거의 실패한다. |
| HepG2/C3A 제외(48개 학습) | 모든 지표가 소수점 넷째 자리 이내로 같다. 간 proxy가 학습에 있느냐는 결과를 바꾸지 않는다. |
| 판정 | 단순 LOCO 수치(a-W)는 **well 기술 효과의 상한**이지 세포주 간 약물 반응 전이 성능이 아니다. 최신 모델은 plate 분리 기준(aP)과 미관측 약물 기준(b)에서 아래 §7의 수치를 넘어야 한다. |

## 1. 범위와 산출물

- **실측:** A1 pseudobulk(50개 세포주 전체), Δ 행렬, LOCO 조건 (a-W)/(a-P)/(b)/(c), 학습 세트 2종(49개, HepG2/C3A 제외 48개), DMSO well 음성 대조, drug 단위 paired bootstrap.
- **미실행:** chemCPA/State/PIE 등 화학 모델 학습, 세포 수준(A2/A3) 추출, EmeraldBay 외부 확인, 용량 외삽.
- **보고서용 파일 (`reports/feedback/R3/`):**
  - [`r3_summary.tsv`](R3/r3_summary.tsv): 조건 × 학습 세트 × 층 × 예측기 평균 지표.
  - [`r3_bootstrap.tsv`](R3/r3_bootstrap.tsv): no-change 및 최선 비공유 기준선 대비 paired bootstrap.
  - [`r3_unit_metrics.tsv.gz`](R3/r3_unit_metrics.tsv.gz): 단위별 지표.
  - [`r3_breakdown.tsv`](R3/r3_breakdown.tsv): MoA·용량·강도별 분해.
  - [`r3_b_vs_c.tsv`](R3/r3_b_vs_c.tsv): 교차 세포주 대 세포주 내부 비교.
  - [`r3_dmso_well_control.tsv`](R3/r3_dmso_well_control.tsv): DMSO well 음성 대조.
  - [`r3_selection.json`](R3/r3_selection.json): 선택된 하이퍼파라미터, 보정 계수, 최근접 세포주, 패널, DEG null 보정.
  - [`r3_test_drugs.tsv`](R3/r3_test_drugs.tsv): 보류 약물 76개.
  - [`pseudobulk_build_record.json`](R3/pseudobulk_build_record.json), [`prep_meta.json`](R3/prep_meta.json), [`line_summary.tsv`](R3/line_summary.tsv), [`r3_run_record.json`](R3/r3_run_record.json).
- **코드:** `src/feedback/r3_pseudobulk_all.py`, `r3_prep.py`, `r3_loco.py`, `r3_report_tables.py`, `experiments/feedback/r3_tahoe.sbatch`.

## 2. A1 pseudobulk 구축 (provenance)

| 항목 | 내용 |
|---|---|
| 원본 | `/mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M/data/train-*.parquet` 3,388개(소유자 hg, 읽기 전용). 쓰기는 하지 않았다. |
| 셀 인코딩 | `genes` = token id, `expressions` = 원 UMI count. 모든 세포는 특수 토큰 1개(id 1, 값 −2)로 시작한다. 모든 샤드에서 이를 확인한 뒤 제거했다. 유전자 token 3–62,712는 `gene_metadata.token_id`와 대응한다. 50개 샤드마다 1개씩 표본 검사했을 때 비정수 값과 0 이하 값은 없었다. |
| 단위 | (sample=well, cell_line_id). 한 well에 50개 세포주가 함께 들어 있다(MOSAIC). plate마다 DMSO_TF well이 2개 있다. |
| 누적량 | 원 count 합, 세포별 CP10k(세포 자신의 총 count 기준) 합, 세포 수, 총 UMI. 저장은 float32다(원 합은 2²⁴ 미만임을 assert로 확인해 정수로 정확하다). |
| QC 세포 집합 | Tahoe `pass_filter=full`. 샤드 행 수 95,624,334는 obs_metadata의 full 세포 수와 같다. plate별 minimal barcode 집합으로 대조하니 샤드 안의 minimal 세포는 **0개**였다. 그룹 67,200개 모두 세포 수가 obs_metadata와 일치했다(불일치 0). 세포가 있는 그룹은 67,018개다. |
| 교차 검증 | SNU-423 1,344 well의 원 count 합이 F4의 `assets/f4/snu423_pseudobulk.npz`(by_cell_line h5ad에서 독립 추출)와 유전자 62,713개 전부 **차이 0**이고 세포 수 불일치도 0이다. |
| 실행 | job **20354**, c3, `--cpus-per-task=48 --mem=400G`(파티션 상한 48 CPU). plate 14개 × 샤드 부분집합 3개 = worker 42개로 처리했다. 벽시계 18:05, CPU 3,128%, worker user 6.16 CPU-h였다. 단일 프로세스 최대 RSS는 18.0 GB(부모와 COW 공유 페이지 포함)이고 main은 9.0 GB였다. sacct DB 접속 실패로 job 전체 메모리는 기록하지 못했다. F4의 추정(16 worker 20–40분)과 비슷하다. |
| 저장 | `assets/r3/pb/` 32 GB(plate마다 `plate_<p>_raw.npy`, `plate_<p>_norm.npy` [96 sample × 50 세포주, 62,713], `groups.tsv`, `genes.tsv`, `build_record.json`). |
| Δ 구성 (job 20419, 1:05) | 발현 E = log1p(세포 평균 CP10k). 대조는 같은 plate DMSO 2 well의 세포를 합친 것이다. Δ = E(well, 세포주) − E(plate DMSO, 세포주)이고, 세포 20개 미만인 (세포주, well)은 NaN으로 두었다(전체의 3.5%). 후보 유전자는 비간 48개 세포주 DMSO 평균 CP10k ≥ 0.1인 **11,132개**다(SNU-423·HepG2 미사용). 저장: `assets/r3/prep/` 2.9 GB. |

## 3. 설계

### 3-1. 보류·가시 정보

| 조건 | 평가 단위 | 학습(가시) 정보 | SNU-423에서 보이는 것 | 질문 |
|---|---|---|---|---|
| **a-W** 미관측 세포주·관측 drug×dose (Tahoe-LOCO) | drug×dose 1,132개 | 학습 세포주의 **모든 well** Δ(시험 well 포함) | DMSO만(전체 프로필 + plate DMSO) | 일반적인 "Tahoe에서 SNU-423만 빼고 학습" 설정 |
| **a-P** 같은 질문, plate 분리 | SNU-423 well 1,301개 | 시험 well의 plate **이외** plate의 학습 세포주 Δ | 동일 | well·plate 기술 효과를 공유하지 않을 때의 전이 |
| **b** 미관측 세포주 + 미관측 약물 | 시험 약물 drug×dose 226개 | 학습 세포주의 학습 약물(80%) | 동일 | 약물 구조 등에서 새 약물을 예측 |
| **c** SNU-423 내부 미관측 약물 (참고) | (b)와 같은 226개 | SNU-423 자신의 학습 약물 조건 | SNU-423 학습 약물 처리 well | 세포주 내 난이도 기준 |

- **시험 세포주:** SNU-423(CVCL_0366). SNU-423 처리 well은 모든 조건에서 학습·선택·패널 정의에 쓰지 않았다. SNU-423 DMSO는 최근접 세포주 선택과 ridge의 세포주 특징에만 쓴다("DMSO 가시" regime). (c)만 SNU-423 학습 약물 처리 well을 쓴다.
- **학습 세트:** tr49(나머지 49개), tr48(HepG2/C3A도 제외, 간 proxy 민감도).
- **약물 분할(b/c):** 약물 379개 중 76개(20%)를 seed 20261110으로 무작위 선택하고 세 용량을 모두 보류했다. plate 대조 약물 Adagrasib은 학습 쪽에 고정했다. 분할에는 데이터를 쓰지 않았다.
- **유전자 패널:** 학습 세트별로 학습 세포주 × 학습 약물 well의 Δ 분산 상위 2,000개다. 패널의 98%가 SNU-423 DMSO에서도 CP10k ≥ 0.1이다. 상위에는 7SK, MT-RNR1, MT2A, MT-ND1, FTL, EGR1 같은 고발현·미토콘드리아 유전자가 많다(§6 한계).
- **정답:** SNU-423 Δ. 세포 50개 이상인 well만 쓴다(drug×dose 1,132/1,137개, well 1,301/1,316개). 여러 well이 있는 조건은 well 평균이다.

### 3-2. 예측기와 정보 등급

| 등급 | 뜻 | 예측기 |
|---|---|---|
| none | 정보 없음 | `no_change` |
| train | 학습 세포주만 쓰고 시험 well은 쓰지 않음 | `global_mean`, aP의 `*_offplate`, b의 `dose_mean`/`moa_dose_mean`/`fp_kernel_ridge`/`fp_knn` |
| train_snu | (c) SNU-423 학습 약물만 | `*_snu` |
| **plate** (진단) | 시험 well과 같은 plate의 DMSO 기준을 공유(well은 공유하지 않음) | `plate_mean_*` |
| **W** (오라클 유사) | 시험 well 안의 학습 세포주 세포를 씀 | a-W의 모든 같은 drug×dose 예측기, `same_well_lines_mean` |
| snu_treated | 다른 plate의 SNU-423 반복 well(잡음 상한 참고) | `snu423_replicate_offplate` |

- **같은 drug×dose 평균** (`same_dd_mean`): 학습 세포주 Δ의 평균이다.
- **크기 보정** (`*_calibrated`): a·예측 + b·global. a, b는 학습 세포주 leave-one-line-out 최소제곱으로만 구했다. SNU-423 정보는 쓰지 않았다.
  - a-W: a = 0.974, b = 0.016으로 사실상 보정이 없다. 같은 well 정보가 크기를 이미 맞추기 때문이다.
  - a-P: a = 0.533, b = 0.408(off-plate 정보를 반쯤 축소).
- **최근접/kNN 세포주:** 학습 세포주 DMSO 프로필의 고분산 유전자 2,000개를 학습 세포주 평균으로 중심화한 뒤 Pearson으로 골랐다. k는 학습 세포주 LOSO로 정했다(k = 20).
  - SNU-423에 가장 가까운 세포주는 RPMI-7951 0.44, HS-578T 0.44, SW 1088 0.42였다. 간 세포주 HepG2/C3A는 상위 10위 안에 없다.
- **ridge_line_pca:** Δ[l, c] ≈ (e_c ⊗ [1, z_l])B 형태의 쌍선형 ridge다.
  - e_c는 학습 세포주 평균 반응의 PCA 점수, z_l은 DMSO PCA다.
  - K, J, λ는 학습 세포주 5-fold(세포주 단위) CV로 골랐다(K = 100, J = 3, λ = 10³). K = 100은 격자 경계다.
- **Morgan 지문 모델:** r = 2, 2,048 bit(rdkit 2026.9.1)이며, 약물 375/379개의 SMILES를 파싱했다.
  - `fp_kernel_ridge`: Tanimoto 커널 × 용량 커널. γ, λ는 학습 약물의 drug 단위 5-fold CV로 골랐다(γ = 0, λ = 3).
  - `fp_knn`: Tanimoto 최근접 약물(k = 20).
  - 시험 약물과 가장 가까운 학습 약물의 Tanimoto는 중앙값 0.30이다. 구조적으로 가까운 약물이 드물다.

### 3-3. 지표·층·불확실성

- **기본 지표:** 패널 2,000 유전자에서 계산했다. RMSE, Pearson, **섭동 중심 r**(평가 단위 전체의 평균 정답·평균 예측을 각각 뺀 Pearson), **plate 중심 r**(같은 plate 단위끼리 중심화해 공유 DMSO 기준 성분을 제거, 그룹당 단위 ≥ 5), retrieval rank(0이 최선, 0.5가 무작위, 중심화 프로필).
- **DEG 정답(자체 정의):** SNU-423의 같은 plate DMSO well 쌍 14개에서 u = (E_a − E_b)/√(1/n_a + 1/n_b)를 구했다.
  - SNU-423 DMSO 발현을 40개 구간으로 나누고, 구간별 |u|의 99.9 분위 τ를 잡았다.
  - 처리 단위는 |Δ_g| > τ·√h(h = Σ(1/n_well + 1/n_DMSO pool)/k²)이면 DEG로 부른다.
  - leave-one-pair-out 위양성률은 평균 0.13%, 최대 0.55%다.
  - 조건당 DEG(패널 내) 중앙값은 28개다(검출 가능 71, 검출 불가 18).
  - **Tahoe 공식 DE는 필터나 정답으로 쓰지 않았다**(F4: null 미보정).
  - 지표: DEG AUPRC(|예측| 순위), precision@k(k = 실제 DEG 수), DEG 부호 일치. DEG가 5개 이상인 단위만 계산했다.
- **층:** F4의 SNU-423 효과 강도 플래그를 썼다. detectable은 R > 모든 DMSO well 쌍, strong은 2배다. 조건 단위에서는 유효 well 중 하나라도 해당하면 그 층에 넣었다(조건 393개 detectable, 90개 strong). 이 층은 시험 정답에서 모델과 무관하게 정의되며 평가 층화에만 쓴다.
- **Bootstrap:** **약물 단위** 군집 paired bootstrap 2,000회, (i) no-change 대비와 (ii) 최선 비공유 기준선(등급 none/train/train_snu 중 해당 조건 "all" 층 RMSE 최저) 대비로 계산했다. drug×dose와 well은 독립 생물학적 반복이 아니다. 같은 실험 안의 다른 well이고 용량은 plate와 교란된다. 따라서 구간은 관측된 약물 집합에 대한 기술적 요약이다.

## 4. 결과 (tr49; tr48은 §4-5)

### 4-1. 음성 대조: DMSO well의 well 공유 신호

| 지표 (DMSO well 28개, SNU-423 DMSO well − plate DMSO pool vs 같은 DMSO well의 학습 세포주 평균) | 값 |
|---|---|
| Pearson 중앙값 (최소–최대) | **0.76** (0.64–0.88) |
| RMSE: no-change → 같은 well 평균 | 28/28 well에서 감소 (예: plate2 0.031 → 0.016, plate6 0.055 → 0.027) |

약물이 없는 well에서도 다른 세포주들의 같은 well 편차가 SNU-423의 편차를 r ≈ 0.76으로 예측한다. 이것은 MOSAIC well 안의 공통 기술 효과다. 예를 들어 ambient RNA, well 단위 처리·라이브러리 차이, plate DMSO 기준의 잡음이 해당한다. 따라서 같은 well을 쓰는 예측기의 성능은 생물학적 반응 전이와 분리되지 않는다.

### 4-2. 조건 (a) 미관측 세포주·관측 drug×dose

**a-W (Tahoe-LOCO, 같은 well 포함):** 단위 1,132개, 약물 379개.

| 예측기 | 등급 | RMSE all / detectable / strong | ΔRMSE vs no-change (all) [95% CI] | 섭동 중심 r all / det | plate 중심 r | DEG AUPRC all / det |
|---|---|---|---|---|---|---|
| no_change | none | 0.1342 / 0.1570 / 0.2215 | – | – | – | – |
| global_mean | train | 0.1241 / 0.1443 / 0.2094 | −0.0101 [−0.0110, −0.0092] | – | – | 0.161 / 0.185 |
| same_dd_mean | W | 0.0798 / 0.0904 / 0.1259 | −0.0544 [−0.0568, −0.0521] | 0.770 / 0.797 | 0.780 | 0.332 / 0.396 |
| same_dd_calibrated | W | 0.0800 / 0.0907 / 0.1265 | −0.0542 | 0.770 / 0.797 | 0.780 | 0.332 / 0.396 |
| nearest_line_same_dd (RPMI-7951) | W | 0.1014 / 0.1104 / 0.1482 | −0.0328 | 0.660 / 0.707 | 0.670 | 0.250 / 0.311 |
| **knn20_same_dd** | W | **0.0753** / 0.0830 / 0.1141 | −0.0588 [−0.0616, −0.0562] | 0.787 / 0.820 | 0.786 | 0.328 / 0.405 |
| ridge_line_pca | W | 0.0774 / 0.0856 / 0.1168 | −0.0568 | 0.782 / 0.811 | 0.779 | 0.312 / 0.390 |

- a-W에서 well을 공유하지 않는 최선 기준선은 `global_mean`이다(ΔRMSE −0.010). well 공유 예측기는 모두 retrieval rank가 0.000으로, 자기 well을 거의 완벽히 찾는다. 반응이 검출되지 않는 739개 조건에서도 `same_dd_mean`의 섭동 중심 r은 0.756이다.
- kNN·ridge가 단순 평균보다 RMSE를 0.0045 줄인다. 하지만 같은 well 정보 안에서의 개선이라 "세포주 특이적 전이"의 증거로 쓸 수 없다.

**a-P (plate 분리):** SNU-423 well 1,301개.

off-plate 정보의 출처는 다음과 같다.

- 같은 drug×dose 313 well(그중 Adagrasib 0.05 µM 대조가 다수).
- 같은 약물의 다른 용량·다른 plate 952 well.
- 약물이 plate 1개에만 있어 전역 평균으로 대체한 36 well.

| 예측기 | 등급 | RMSE all / det / strong | ΔRMSE vs no-change all [95% CI] | det [95% CI] | 섭동 중심 r all / det | plate 중심 r all / det |
|---|---|---|---|---|---|---|
| no_change | none | 0.1356 / 0.1639 / 0.2359 | – | – | – | – |
| global_offplate | train | 0.1275 / 0.1523 / 0.2256 | −0.0081 | −0.0116 | (−0.39, plate 산물) | – |
| same_drug_offplate | train | 0.1200 / **0.1406** / 0.1971 | −0.0156 [−0.0180, −0.0132] | −0.0233 [−0.0290, −0.0183] | **0.380 / 0.410** | **0.464 / 0.503** |
| **hier_offplate_calibrated** | train | **0.1188** / 0.1412 / 0.2014 | **−0.0168 [−0.0187, −0.0150]** | −0.0227 [−0.0266, −0.0194] | 0.340 / 0.358 | 0.441 / 0.469 |
| knn20_hier_offplate | train | 0.1239 / 0.1427 / **0.1919** | −0.0117 | −0.0212 | 0.376 / 0.381 | 0.437 / 0.471 |
| plate_mean_otherdrugs | **plate** | 0.1155 / 0.1414 / 0.2148 | −0.0202 | −0.0225 | 0.378 / 0.363 | (−0.74, 구성상 무의미) |
| same_well_lines_mean | **W** | 0.0812 / 0.0946 / 0.1343 | −0.0544 | −0.0692 | 0.767 / 0.795 | 0.779 / 0.810 |

- **최선 비공유 기준선 대비 well 공유 오라클:**
  - RMSE −0.0377 [−0.0399, −0.0354], 섭동 중심 r +0.43 [+0.40, +0.45].
  - well 공유 정보가 비공유 정보보다 RMSE 개선폭을 3배 이상 크게 만든다.
- **plate만 공유해도 유리하다:** 같은 plate 다른 약물의 평균(`plate_mean_otherdrugs`)이 RMSE에서 비공유 최선보다 0.0034 낫다([−0.0055, −0.0013]). Δ 정답에 plate DMSO 기준 잡음이 상당 부분 들어 있다는 뜻이다. plate 중심 r은 이 성분을 빼며, `same_drug_offplate`의 r을 0.38에서 0.46으로 올린다.
- **strong 층**에서는 off-plate 같은 약물 정보가 실질적이다: kNN −0.044 [−0.065, −0.024], 섭동 중심 r 0.43.
- **잡음 상한 (aP_repsubset):** 다른 plate에 SNU-423 반복 well이 있는 309 well이다.
  - SNU-423 자신의 off-plate 반복 well은 RMSE 0.1310으로 no-change(0.1275)보다 나쁘다.
  - 학습 세포주의 같은 약물 off-plate 평균(0.1124)이 오히려 낫다(+0.0186 [+0.0118, +0.0245]).
  - well 하나짜리 SNU-423 정답은 같은 drug×dose를 다른 plate에서 다시 재도 잘 재현되지 않는다(F4의 반복 r 0.41과 일치).
  - 따라서 plate 분리 평가에서 도달 가능한 상한 자체가 낮다.

### 4-3. 조건 (b) 미관측 세포주 + 미관측 약물 / (c) SNU-423 내부 미관측 약물

같은 시험 단위 226개(약물 76개)에서 비교했다.

| 예측기 | 등급 | RMSE all / det / strong | ΔRMSE vs no-change all [95% CI] | 섭동 중심 r all / det | plate 중심 r all | retrieval |
|---|---|---|---|---|---|---|
| no_change | none | 0.1292 / 0.1450 / 0.1811 | – | – | – | – |
| (b) global_mean | train | 0.1201 / 0.1337 / 0.1707 | −0.0091 | – | – | – |
| (b) **dose_mean** | train | **0.1185** / 0.1322 / 0.1675 | **−0.0107 [−0.0130, −0.0085]** | 0.162 / 0.129 | 0.119 | 0.378 |
| (b) moa_dose_mean | train | 0.1212 / 0.1358 / 0.1695 | −0.0079 | 0.153 / 0.122 | 0.024 | 0.361 |
| (b) fp_kernel_ridge | train | 0.1187 / **0.1315** / **0.1641** | −0.0105 | 0.146 / 0.153 | 0.035 | 0.375 |
| (b) fp_knn | train | 0.1205 / 0.1336 / 0.1674 | −0.0087 | 0.107 / 0.109 | 0.012 | 0.392 |
| (b) plate_mean_traindrugs | plate | 0.1095 / 0.1229 / 0.1597 | −0.0197 | 0.386 / 0.392 | – | 0.130 |
| (b) same_well_lines_mean | W | 0.0760 / 0.0824 / 0.1051 | −0.0532 | 0.781 / 0.807 | 0.790 | 0.000 |
| (c) global_mean_snu | train_snu | 0.1170 / 0.1288 / 0.1654 | −0.0122 | – | – | – |
| (c) **dose_mean_snu** | train_snu | **0.1148** / 0.1270 / 0.1617 | **−0.0144 [−0.0173, −0.0112]** | 0.190 / 0.163 | 0.115 | 0.346 |
| (c) fp_kernel_ridge_snu | train_snu | 0.1151 / **0.1259** / **0.1568** | −0.0140 | 0.176 / 0.189 | 0.037 | 0.334 |
| (c) plate_mean_traindrugs_snu | plate | 0.0974 / 0.1109 / 0.1486 | −0.0318 | 0.530 / 0.504 | – | 0.081 |

- **지문 ridge의 이득:** (b)에서 `dose_mean` 대비 all +0.0002 [−0.0005, +0.0010], detectable −0.0007 [−0.0022, +0.0007], strong −0.0034 [−0.0068, +0.0002]다. 강한 반응에서만 약한 경향이 있고 유의하지 않다. plate 중심 r 0.035는 plate 효과를 빼면 약물 특이적 구조가 거의 남지 않음을 뜻한다. `dose_mean`의 섭동 중심 r(0.16)도 상당 부분 용량–plate 교란에서 온다.
- **교차(b) vs 내부(c)** ([`r3_b_vs_c.tsv`](R3/r3_b_vs_c.tsv), 같은 단위의 paired bootstrap):

  | 비교 | 지표 | 차이 (b − c) [95% CI] |
  |---|---|---|
  | dose_mean | RMSE | +0.0037 [+0.0025, +0.0048] |
  | dose_mean | 섭동 중심 r | −0.027 [−0.034, −0.020] |
  | fp_kernel_ridge (detectable) | RMSE | +0.0056 [+0.0040, +0.0070] |

  SNU-423 자신의 다른 약물 반응이 다른 세포주들의 같은 정보보다 조금 낫다. 다만 차이(RMSE 0.004–0.006)는 no-change 대비 이득(0.011–0.014)의 1/3 수준이다. 두 경우 모두 "새 약물"이 어려움의 대부분을 차지한다.
- (a-P)의 off-plate 같은 약물 정보(det 섭동 중심 r 0.41)와 (b)/(c)의 미관측 약물(≤ 0.19)을 비교하면, **세포주 간 전이보다 약물 정보 유무가 더 큰 요인**이다. F3의 유전적 LOCO에서 "새 표적"이 핵심 난점이었던 것과 같은 양상이다.

### 4-4. 효과 강도·MoA·용량별 실패 패턴 (a-P, tr49)

- **강도별 이득:** no-change 대비 `same_drug_offplate` RMSE 이득은 not detectable 0.012, detectable(not strong) 0.019, strong 0.039다. 같은 well 오라클은 0.048, 0.060, 0.102다. 그런데 not detectable에서도 well 오라클 이득이 0.048이다. 이는 반응이 없는 조건에서 well 기술 효과가 이득의 대부분임을 보여준다.
- **효과 크기와 상관:** F4 R과 섭동 중심 r의 Spearman은 off-plate 같은 약물 0.13, same-well 0.40이다.
- **off-plate가 no-change보다 나쁜 비율:** 전체 well의 21%다(strong 21%, not detectable 23%).
- **strong well 92개에서 off-plate 같은 약물 r이 높은 예:** 넓은 기전의 강한 교란이다. Panobinostat/Tucidinostat(HDAC), Bortezomib/Ixazomib(proteasome), Homoharringtonine/Harringtonine(번역), Ouabain/Digitoxin(Na⁺/K⁺-ATPase), Dinaciclib(CDK)의 r이 0.70–0.85다.
- **같은 strong well에서 r이 0 이하인 예:** mTOR/PI3K 계열인 Sapanisertib 0.5/5 µM(−0.62, −0.34), GSK1059615, Bimiralisib가 있다. 그 밖에 Paclitaxel 0.5 µM(−0.27), ML264, Vemurafenib, Epirubicin 5 µM이 있다.
  - 이것이 세포주 특이 반응인지, 서로 다른 plate·용량에서 측정된 탓인지는 이 설계로 구분할 수 없다.
  - **HCC 관련 약물 Regorafenib 0.5 µM은 r 0.02**로 거의 예측되지 않는다. Sorafenib과 Lenvatinib은 Tahoe 약물 목록에 없다.
- **MoA-broad별 off-plate r:** inhibitor/antagonist 0.36, unclear 0.43, activator/agonist 0.47이다. same-well은 모든 군에서 약 0.76으로 차이가 없어, 이것이 기술 효과임을 다시 보여준다.
- **용량별 RMSE 이득(off-plate 같은 약물):** 0.05 µM 0.013, 0.5 µM 0.013, 5 µM 0.020이다. **용량–반응 해석은 하지 않는다**(용량이 plate와 교란됨).

### 4-5. HepG2/C3A 제외 민감도 (tr48)

| 조건 / 예측기 | tr49 RMSE | tr48 RMSE | tr49 섭동 중심 r | tr48 |
|---|---|---|---|---|
| a-W knn20_same_dd | 0.0753 | 0.0756 | 0.787 | 0.786 |
| a-P hier_offplate_calibrated | 0.1188 | 0.1188 | 0.340 | 0.339 |
| a-P same_drug_offplate | 0.1200 | 0.1201 | 0.380 | 0.379 |
| b dose_mean | 0.1185 | 0.1184 | 0.162 | 0.162 |
| c dose_mean_snu | 0.1148 | 0.1147 | 0.190 | 0.189 |

HepG2/C3A는 SNU-423의 DMSO 최근접 상위 10위에 없고, 학습에서 빼도 결과가 변하지 않는다. "간 유래 세포주를 학습에 넣으면 SNU-423 예측이 좋아진다"는 근거는 이 기준선 수준에서는 없다.

## 5. 해석

1. **Tahoe-100M에서의 단순 LOCO는 well 공유 누출을 내장한다.** drug×dose 대부분이 well 하나이고 50개 세포주가 그 well을 공유한다. 따라서 "학습에서 본 drug×dose"는 "같은 well의 다른 세포주 세포"와 같다. DMSO 음성 대조(r 0.76)가 보여주듯 이 정보는 약물과 무관하게 SNU-423 Δ를 예측한다. Tahoe로 학습한 모델이 SNU-423만 빼고 학습해 높은 a-W 점수를 얻는다면, 이 기준선(RMSE 0.075, 섭동 중심 r 0.79)을 넘어야 비로소 의미가 생긴다.
2. **well을 공유하지 않으면 전이 신호는 작지만 0은 아니다.** 같은 약물(대개 다른 용량, 다른 plate)의 학습 세포주 평균은 no-change 대비 RMSE를 12% 줄인다(검출 가능 조건 14%). 섭동 중심 r은 0.38–0.41이고 strong 조건에서 0.43이다. 강한 넓은 기전 약물에서 주로 성공하고, 경로 특이적 억제제(mTOR/PI3K, MAPK 일부)에서 실패한다.
3. **새 약물은 교차·내부 모두 거의 실패한다.** 지문 기반 kernel ridge·kNN은 용량별 평균을 넘지 못한다. 그 용량별 평균의 약한 상관도 plate 교란 성분이다.
4. **크기 보정은 이 자료에서 큰 의미가 없다.** a-W의 보정 계수는 ≈ 1, a-P에서는 축소(0.53)가 일부 도움이 되지만 strong에서는 오히려 손해다. F3(유전적)와 달리 학습 세포주와 SNU-423의 반응 크기 차이가 작다.

## 6. 한계

- **세포주 모델이다.** SNU-423은 성인 HCC 유래 **세포주 1개**이고 MOSAIC 50개 세포주 공동배양, 단일 시점(프리프린트 기준 24 h, 데이터로 미확인)이다. 환자 HCC 약효·예후나 HCC 일반화로 확장할 수 없다.
- **well 공유:** 학습 세포주와 SNU-423이 같은 well·같은 plate DMSO를 공유한다. 기술 효과와 공유 생물학적 반응을 분리할 수 없다. plate 분리(a-P)도 같은 실험(같은 라이브러리 처리 체계) 안이다.
- **plate·용량 교란:** 약물 379개 중 354개가 용량마다 다른 plate에 놓여 있다. a-P의 off-plate 정보는 대부분 "다른 용량"이고, `dose_mean`의 상관도 plate 성분을 포함한다. **용량–반응 주장을 하지 않는다.**
- **정답 잡음:** 처리 well의 31.5%만 DMSO well 쌍 null을 넘는다(F4). 다른 plate의 SNU-423 반복 well이 no-change보다 나쁠 정도로 단일 well 정답의 재현성이 낮다. 전체 평균 지표는 잡음이 지배하므로 detectable/strong 층을 함께 보아야 한다.
- **Tahoe 공식 DE 미사용:** 반응 필터·DEG 정답으로 쓰지 않았다(F4: DMSO 대 DMSO null 없음). 자체 DEG 정의는 DMSO well 쌍 14개에 기반하고, 셀 수 스케일링(√h)이 well 효과에는 정확하지 않다. 검출 불가 조건에서도 DEG 중앙값이 18개(기대 위양성 ≈ 2)이므로 DEG 지표는 보수적으로 해석해야 한다.
- **패널:** 분산 기준 상위 2,000개에 미토콘드리아·7SK·고발현 유전자가 많아 기술 변동에 민감하다. 대안 패널(예: MT/rRNA 제외, 평균–분산 보정)은 실행하지 않았다.
- **불확실성:** drug 단위 bootstrap은 독립 생물학적 반복을 가정하지 않는 기술적 요약이다. 하이퍼파라미터는 학습 세포주에서만 골랐다. 다만 ridge K = 100은 격자 경계다.
- **기타:**
  - 약물 4개는 SMILES를 파싱하지 못해 지문이 0 벡터다.
  - Adagrasib 0.05 µM(16 well) 대조가 a-P의 "같은 drug×dose off-plate" 출처의 다수를 차지한다.
  - 같은 공개 자료를 반복 사용하므로 blind test가 아니다.

## 7. 최신 화학 모델이 넘어야 할 기준

이 기준은 PIE Tahoe recipe, State, chemCPA를 **SNU-423 처리 세포를 모두 빼고 재학습**한 경우에 적용한다.

| 평가 설정 | 넘어야 할 최선 단순 기준선 (tr49) | 오라클/진단 참고 |
|---|---|---|
| a-W (Tahoe-LOCO, 같은 well 학습 포함) | knn20_same_dd: RMSE **0.0753**, 섭동 중심 r **0.787**, retrieval 0.000, DEG AUPRC 0.328 (detectable 0.0830 / 0.820 / 0.405). 이 점수는 well 공유 상한이며, 모델이 이 설정에서 이 값을 넘지 못하면 같은 well 평균보다 못한 것이다. | DMSO well 음성 대조 r 0.76 |
| a-P (plate 분리) | RMSE **0.1188** (hier_offplate_calibrated); 섭동 중심 r **0.380**, plate 중심 r **0.464** (same_drug_offplate); detectable RMSE 0.1406 / r 0.410; strong RMSE 0.1919 (kNN) / r 0.435 | same-well 0.0812 / 0.767; plate_mean 0.1155 |
| b (미관측 세포주·약물) | RMSE **0.1185** (dose_mean), detectable 0.1315 / strong 0.1641 (fp_kernel_ridge); 섭동 중심 r 0.162, plate 중심 r 0.119 | same-well 0.0760; plate_mean 0.1095 |
| c 대비 | SNU-423 내부 dose_mean_snu 0.1148: 교차 모델이 이를 넘으면 "다른 세포주 정보가 SNU-423 자체 정보보다 낫다"고 말할 수 있다 | – |

**필수 조건:**

1. Tahoe-100M으로 사전학습된 공개 checkpoint(예: State의 Tahoe 학습본, Tahoe 기반 PIE 자원)는 SNU-423을 보았을 가능성이 높다. 학습 데이터 범위를 확인하기 전에는 SNU-423을 "미관측"이라고 부르지 않는다(이번 작업에서 확인하지 않음).
2. 모델 평가는 a-W와 a-P를 **둘 다** 보고하고, 같은 well 오라클과의 차이를 함께 적는다.
3. 반응 강도(F4 detectable/strong)별로 층화한다.
4. 약물 단위 불확실성을 쓴다.
5. 용량은 범주형으로만 다룬다.
6. 하이퍼파라미터는 학습 세포주(또는 HepG2/C3A 등 검증 세포주)로만 고른다.

## 8. 수행 기록

| job | 단계 | 노드 / 자원 | 벽시계 · MaxRSS(단일 프로세스) | 상태 |
|---|---|---|---|---|
| 20341 | pb_test | c4, 16 CPU / 160 GB | 0:37 | 실패: 샤드가 plate 2개에 걸침 → 수정 |
| 20345, 20348, 20349 | pb_test (plate 2개 × 3–8 샤드) | c4, 16 / 160 GB | 0:56–1:50 · 13–17 GB | 완료, 속도 개선 반복(직접 CSR, float32 누적) |
| **20354** | **pb_all (A1)** | **c3, 48 CPU / 400 GB, worker 42** | **18:05 · 18.0 GB (main 9.0 GB), worker 6.16 CPU-h** | **완료** |
| 20419 | prep (Δ) | c4, 8 / 96 GB | 1:05 · 4.8 GB | 완료 |
| 20420 | loco 시험(tr49, nboot 50) | c4, 16 / 96 GB | 8:24 · 7.3 GB | 완료, 대체됨 |
| 20446 | loco 전체 | c5, 16 / 96 GB | 17:55 · 8.2 GB | 실패: bootstrap 색인 오류 → 수정 |
| **20463** | **loco 최종 (tr49 + tr48, nboot 2,000)** | **c3, 16 / 96 GB** | **15:51 · 8.3 GB** | **완료** |
| 20464, 20476 | srun: 요약 재계산 시험, 표·b-vs-c 생성 | 2 / 16 GB | 수 초–수십 초 | 완료 |

- 모든 job은 `cpu` 파티션 c3–c5에서 돌렸다(`--exclude=g1,g2,g3,g4`). sacct DB(c1:6819) 접속 실패로 job 전체 메모리는 기록하지 못했다. 위 값은 `/usr/bin/time -v`로 잰 것이다.
- **환경:** 새 venv `assets/envs/r3`(numpy 1.26.4, pandas 2.2.3, pyarrow 17.0.0, scipy 1.12.0, scikit-learn 1.5.2, rdkit 2026.9.1). 기존 환경은 건드리지 않았다.
- **G1 자산:**
  - `assets/r3/pb`(32 GB)
  - `assets/r3/prep`(2.9 GB)
  - `assets/r3/loco`(278 MB, 조건·학습 세트별 정답과 모든 예측 npz, 패널 token id)
- 다운로드는 없다. 원본 미러에는 쓰지 않았다.

## 9. 다음 단계와 미해결

1. **well 효과 분리:** DMSO well의 세포주 공통 편차를 추정해 "well 보정 Δ"를 정의하는 정답 변형을 만든다. 예: 같은 well의 50개 세포주 공통 성분 회귀. 단, 공유 생물학적 반응도 함께 제거될 위험이 있다. 그 위에서 a-W와 a-P의 차이가 줄어드는지 확인한다.
2. **패널 민감도:** MT/rRNA/7SK 제외, 평균–분산 보정 패널.
3. **세포 수준 자료(A2/A3) 추출:** chemCPA/State 재학습용으로, 이번 pass의 코드로 (세포주, well)당 100개 세포를 뽑으면 된다.
4. **공개 Tahoe 학습 checkpoint의 SNU-423 노출 여부 확인.**
5. **EmeraldBay 외부 확인:** 9개 공유 약물, 다른 시점·용량에서 방향만 비교한다.
