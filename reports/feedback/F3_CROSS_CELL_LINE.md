# F3 — 4개 세포주 CRISPRi 데이터 계약과 LOCO 단순 기준선 (피드백 3, 4-genetic)

**작성:** 2026-11-10 · 브랜치 `research/feedback-20261110` (커밋하지 않음) · 계획: [FEEDBACK_PLAN_20261110.md](../../docs/FEEDBACK_PLAN_20261110.md) §3, §4-1 · 실행: G1 Slurm `cpu` 파티션.

**요약.** K562·RPE1(Replogle 2022)과 HepG2·Jurkat(Nadig 2025, GSE264667)의 원 카운트를 읽어 데이터 계약을 만들고, 4개 LOCO fold에서 단순 기준선을 평가했다. 데이터 계약 판정은 **조건부 적격(conditional)**이다. 4개 세포주 모두 원 정수 카운트, non-targeting 대조군, gem_group 배치 정보, 같은 CRISPRi modality를 갖추고 있다. ≥20 cells 표적의 4-way 교집합은 **1,688개**, 공통 유전자 패널은 **6,642개**다. 다만 day/time 메타데이터가 파일에 없고, HepG2는 이전 연구에서 이미 본 자료이며, 세포주와 실험실·라이브러리·시퀀싱 깊이가 분리되지 않는다.

LOCO 결과를 지표별로 정리하면 다음과 같다.

1. **미관측 세포주·관측 표적(A):** 소스 세포주의 같은 표적 반응을 평균하고 소스 내부 CV로 크기를 보정한 예측(`calibrated_same_target`)이 4개 fold 모두에서 ΔRMSE가 가장 낮았다. no-change 대비 개선량은 −0.0017(K562)~−0.0120(RPE1)이며, 모든 fold에서 bootstrap 95% CI가 0을 넘지 않는다.
2. 섭동 중심(perturbation-centred) Pearson은 0.31–0.35였다. retrieval rank는 0.09–0.12로, 0.5인 무작위보다 낫다. 같은 표적의 반응 *패턴*은 세포주를 넘어 전이된다.
3. 보정하지 않은 같은 표적 평균은 K562와 Jurkat에서 no-change보다 RMSE가 **나빴다**. 소스 세포주의 반응 크기가 더 크기 때문이다.
4. **이중 OOD(Bs, 미관측 표적 + 미관측 세포주):** 어떤 기준선도 일관되게 no-change를 넘지 못했다. K562·Jurkat에서는 no-change가 RMSE 최저였다. RPE1·HepG2에서는 ridge/전체 평균이 공통 반응 덕분에 약간 나았다(−0.0085, −0.0018).
5. **같은 세포주 내 미관측 표적(C):** PCA-ridge가 no-change보다 4개 fold 모두에서 나았다(−0.0030~−0.0160). 같은 표적 집합에서 비교하면 교차 세포주 같은 표적 전이(A)의 섭동 중심 Pearson이 세포주 내 ridge(C)보다 0.10–0.16 높았다.
6. 따라서 이 패널에서 어려움의 본질은 "새 세포주"보다 **"새 표적"**이다. 새 세포주로의 전이는 소스에서 같은 표적을 관측했을 때만 실질적인 신호를 준다.

---

## 1. 데이터 계약 (Step 1, G2 Gate)

파일은 모두 h5py로 obs/var만 읽어 확인했다. 발현 행렬은 Slurm 작업에서 8,000행 블록으로 스트리밍했다. 원 데이터 경로는 읽기 전용이다.

| 항목 | K562 | RPE1 | HepG2 | Jurkat |
|---|---|---|---|---|
| 연구 / 출처 | Replogle 2022 Cell, figshare 20029387 | Replogle 2022, figshare 20029387 | Nadig 2025 Nat Genet, GEO GSE264667 | Nadig 2025, GSE264667 |
| 파일 (md5) | `K562_essential_raw_singlecell_01.h5ad` (4f1122…, figshare 공개값 일치) | `rpe1_raw_singlecell_01.h5ad` (6a2a9d…, 로컬값만) | `GSE264667_hepg2_raw_singlecell_01.h5ad` (dfc676…, 로컬값만) | `GSE264667_jurkat_raw_singlecell_01.h5ad` (dae9fa…, 로컬값만) |
| 세포 유형 | 만성골수성백혈병 | hTERT 불멸화 망막색소상피(비암) | **간모세포종 유래**(성인 HCC 아님) | T-ALL |
| 세포 수 × 유전자 패널 | 310,385 × 8,563 | 247,914 × 8,749 | 145,473 × 9,624 | 262,956 × 8,882 |
| 원 카운트 여부 | 정수, X dense float32 (확인) | 정수 | 정수 | 정수 |
| 유전자 ID | var index = Ensembl `gene_id`, `gene_name` = symbol; 패널은 저자가 필터링(평균 >0.1 UMI/cell, 관측 최소값 0.10) | 동일 | 동일 | 동일 |
| 표적 열 / guide 열 | `gene`(symbol), `gene_id`(Ensembl) / `gene_transcript`, `sgID_AB` (TSS별 dual-sgRNA) | 동일 | 동일 | 동일 |
| 대조군 라벨, 세포 수, guide 수 | `non-targeting`, 10,691, 97 | 11,485, 113 | **4,976**, 130 | 12,013, 130 |
| 배치 열, 수 | `gem_group`, 48 (모두 대조군 포함) | 56 | 56 | 55 |
| gem_group당 대조군 (최소/중앙/최대) | 116 / 227.5 / 281 | 119 / 209.5 / 260 | **5** / 91.5 / 123 | 39 / 224 / 290 |
| day/time | **파일 메타데이터에 없음** | 없음 | 없음 | 없음 |
| UMI/cell 중앙값 (`UMI_count`) | 13,396 | 12,126 | 15,641 | 10,160 |
| 표적 수 (전체 / ≥20 cells) | 2,057 / **2,021** | 2,393 / **2,204** | 2,393 / **2,160** | 2,393 / **2,270** |
| 표적당 세포 (최소/Q1/중앙/Q3/최대) | 5/78/121/181/1,996 | 2/41/72/117/3,580 | 2/30/45/69/1,213 | 1/51/83/126/2,555 |
| 다중 TSS(guide 2개 이상) 표적 | 118 | 135 | 135 | 135 |
| knockdown (계산값; 파일 미제공) 중앙값, ≥50% 비율, <20% 비율 | 0.84, 92.1%, 1.3% | 0.91, 95.2%, 1.2% | 0.90, 94.4%, 1.6% | 0.88, 89.9%, 2.5% |

- **라이브러리 크기:** obs `UMI_count`는 전사체 전체 UMI다. 행렬의 행 합은 그 0.97–0.99배(중앙값)다. 정규화에는 `UMI_count`를 사용했다.
- **knockdown:** 1 − (섭동 세포의 표적 유전자 평균 CP10k) / (대조군 평균)으로 계산했고, 표적 유전자가 해당 세포주 패널에 있을 때만 정의된다.
- **표적 겹침 (≥20 cells):**
  - 4-way: **1,688**.
  - 3-way: K562&RPE1&HepG2 1,740; K562&RPE1&Jurkat 1,805; K562&HepG2&Jurkat 1,764; RPE1&HepG2&Jurkat 1,980.
  - pairwise: K562&RPE1 1,875; K562&HepG2 1,828; K562&Jurkat 1,916; RPE1&HepG2 2,039; RPE1&Jurkat 2,123; HepG2&Jurkat 2,086.
  - 단독 표적: K562 23 / RPE1 4 / HepG2 3 / Jurkat 6.
  - symbol↔Ensembl 충돌: 0.
  - RPE1·HepG2·Jurkat은 같은 2,393-표적 essential 라이브러리를 쓰고, K562 essential 라이브러리는 2,057개다.
- **유전자 패널:** 4-way 교집합은 **6,642**개(Ensembl), pairwise 교집합은 7,226–7,735개다. 측정되는 표적 유전자 수(4-way 패널 내)는 K562 1,745 / RPE1 1,865 / HepG2 1,833 / Jurkat 1,923이다.
- **LOCO 표적 수:**
  - 관측 표적(소스 ≥1개에서 섭동): K562 1,998 / RPE1 2,200 / HepG2 2,157 / Jurkat 2,264.
  - 자연 이중 OOD: 23 / 4 / 3 / 6. 너무 작아서, 소스에서 표적을 제거하는 **모의 이중 OOD**를 주 분석으로 삼았다.
- **판정: 조건부 적격.** 필수 기준은 모두 충족했다: 처리 후 scRNA + 대조군, 원 카운트, 배치 정보, 같은 modality, 4-way 표적 ≥300, 공통 유전자 ≥2,000. 조건은 다음과 같다.
  - (i) day/time은 논문에서 확인해야 하며 이 작업에서는 검증하지 않았다.
  - (ii) HepG2는 이전 V2 프로젝트에서 이미 본 자료다(`manifests/exposure_ledger.tsv`: development_previously_exposed). **새 blind cohort가 아니다.**
  - (iii) 세포주 효과는 실험실·라이브러리 버전·시퀀싱 깊이·screen 설계(Replogle vs Nadig)와 섞여 있다.
  - (iv) HepG2는 대조군이 적고, 대조군이 5개뿐인 gem_group이 있다.
  - 상세: [data_contract.tsv](F3/data_contract.tsv), [data_contract.json](F3/data_contract.json), [target_overlap.tsv](F3/target_overlap.tsv).

## 2. Pseudobulk 및 정답 구성 (Step 2)

- **세포 단위 처리:** CP10k(`UMI_count` 기준) → log1p. 블록 단위로 (세포주, 표적)별 합, 제곱합, 선형 합, 무작위 반쪽(seed 고정)별 합, 표적×gem_group 세포 수, gem_group별 대조군 합을 누적했다. 산출물은 `assets/f3/pseudobulk/<line>.npz`이고, 세포주당 51–83초 걸렸다.
- **Δ(정답):** 표적 평균 − **배치 매칭 대조군 평균**이다. 대조군 평균은 gem_group별 대조군 평균을 그 표적 세포의 gem_group 구성비로 가중해 만들었다. 모든 gem_group에 대조군이 있어 세포주 수준 대조군으로 대체할 필요가 없었다.
- **DEG 정답:** Welch t-test(표적 vs 배치 매칭 대조군 평균, 대조군 분산은 세포주 전체 풀링)로 계산했다. 공통 6,642 유전자에 대해 표적별 BH를 적용하고 FDR < 0.05를 기준으로 했다. 행렬은 `assets/f3/matrices/<line>_delta.npz`에 저장했다(Δ, t, q, DEG, 반쪽 Δ, KD, n).
- **외부 교차검증 (QC 전용):** HKU CRISPRi12 예치 자료(figshare 10.25442/hku.34002315.v1, CC BY-NC-SA 4.0, crispyx v0.1.4)와 비교했다. 파일 md5 8개는 MANIFEST와 모두 일치했다.
  - 우리 Δ와 그들의 batch-aware effect의 표적별 Pearson 중앙값은 **0.9999–1.0000**이다(4개 세포주, 공통 유전자 6,640–6,642).
  - DEG 수 중앙값: 우리 Welch / 그들 van Elteren(batch-stratified) = K562 61/48, RPE1 455/323, HepG2 **82/21**, Jurkat 82/53. Jaccard 중앙값은 0.26–0.56이다.
  - Δ 구성은 독립 구현과 일치한다. DEG 정의는 검정에 민감하며, 특히 HepG2에서 우리 Welch DEG가 많다. HepG2 대조군이 적고 배치 층화를 하지 않았기 때문으로 보인다. 그래서 **HepG2의 DEG 기반 지표(AUPRC/F1)는 보수적으로 해석**해야 한다. 이 자료는 정답이나 모델 입력으로 쓰지 않았다([crosscheck_hku.tsv](F3/crosscheck_hku.tsv)).
- **세포주별 반응 크기** ([line_pseudobulk_summary.tsv](F3/line_pseudobulk_summary.tsv)):

  | 세포주 | DEG 중앙값 | ‖Δ‖₂ 중앙값 |
  |---|---:|---:|
  | K562 | 61 | 3.96 |
  | RPE1 | **455** | **6.82** |
  | HepG2 | 82 | 5.42 |
  | Jurkat | 82 | 5.20 |

  같은 표적 Δ의 세포주 간 Pearson(공통 패널) 중앙값은 0.16–0.25이며, HepG2–RPE1이 0.249로 가장 높다. 주의: 이 파일의 `frac_kd_ge50`은 표적 유전자가 미측정인 표적도 분모에 넣은 값이다. 측정 표적 기준 값은 위 계약 표를 따른다.
- **Fold별 유전자 패널:** 소스 3개 세포주의 Δ만으로 유전자별 분산을 계산해 상위 2,000개를 골랐고, 여기에 측정된 표적 유전자 2,035개(정체성 정보만)를 합쳤다. 크기는 3,121–3,188개다. 보류 세포주의 처리 세포는 패널 정의에 쓰지 않았다(`assets/f3/loco/panel_<L>.npy`).

## 3. 설계 (Step 3, G3 Gate)

각 fold에서 보류 세포주 L의 처리 세포는 학습·선택·패널 정의에 전혀 쓰지 않았다. **L의 대조군만** 볼 수 있으며, 가장 가까운 세포주를 고르는 데만 사용한다.

| 조건 | 학습(가시) 자료 | 평가 표적 | 2×2 위치 |
|---|---|---|---|
| A 미관측 세포주·관측 표적 | 소스 3개의 처리+대조, L 대조 | 소스 ≥1개에서 섭동된 L의 표적 (1,998–2,264) | 미관측 맥락 × 관측 표적 |
| Bn 자연 이중 OOD | 동일 | 소스 어디에도 없는 표적 (3–23) | 참고만 |
| Bs 모의 이중 OOD | 소스에서 S_L 표적의 처리 세포를 **모두 제거** | S_L (L 표적의 무작위 20%, 404–454, seed 20261110) | 미관측 맥락 × 미관측 표적 |
| C 세포주 내 미관측 표적 | L 자체의 나머지 80% 표적 + L 대조 | S_L | 관측 맥락 × 미관측 표적 |
| D split-half 기준 | S_L 표적 세포의 반쪽 0 | S_L 반쪽 1 | 관측 맥락 × 관측 표적 (잡음 기준) |

**예측기:**

- `no_change`: Δ=0.
- `global_mean`: 소스 세포주별 평균 Δ의 평균.
- `same_target_mean`: 표적 t를 가진 소스들의 Δ_t 평균.
- `nearest_line_same_target`: L 대조군 평균 발현과 Pearson이 가장 높은 소스 중 t를 가진 첫 세포주의 Δ_t. 선택된 세포주: K562→Jurkat, RPE1→HepG2, HepG2→RPE1, Jurkat→K562.
- `calibrated_same_target`: a·same_target + b·global. a, b는 소스 내부 leave-one-source-line-out 최소제곱으로 정했고, a=0.41–0.44, b=0.20–0.36이다.
- `ridge_pca`: Ahlmann-Eltze 2025 방식의 쌍선형 ridge. 유전자 임베딩 G는 소스 Δ(유전자×(세포주,표적)) PCA의 상위 K개다. 섭동 임베딩은 표적 유전자의 G 행이므로 미관측 표적도 예측할 수 있다.
  - K∈{5…200}, λ∈{0.1…10⁴}는 소스 내부 LOSO-CV(A·Bs) 또는 L 학습 표적 내 5-fold CV(C)의 RMSE로 골랐다.
  - 선택값: A/Bs는 4개 fold 모두 K=200, λ=100. C는 K562 25/10, RPE1 200/100, HepG2 50/10, Jurkat 200/100. K=200은 격자 경계지만, CV 곡면이 평평하다(최적 대비 최악 ≈ +1%).

**지표:** 모두 표적 유전자 자신은 제외했다.

- ΔRMSE, ΔPearson.
- Systema식 source-centred Pearson: 소스 평균 Δ를 정답·예측 양쪽에서 뺀 값.
- **perturbation-centred Pearson**: 평가 집합에서 예측끼리, 정답끼리의 표적 평균을 각각 뺀 값. 평가 전용이며, 공통 반응만 맞히는 예측은 NaN이 된다.
- 상위 20개 DEG(|t| 순) 방향 일치율.
- DEG AUPRC 및 F1@k(k = 실제 DEG 수).
- retrieval 정규화 rank: 0이 최선, 0.5가 무작위. 비표적 판독 유전자에서 섭동 중심 Pearson으로 계산한다.
- paired bootstrap 2,000회: 표적 단위로 재표집한다. **표적은 독립 생물학적 반복이 아니므로** 구간은 관측 표적 집합에 대한 기술적 요약이다.

## 4. 결과

### 4-1. 조건 A — 미관측 세포주·관측 표적 (모든 관측 표적)

열: RMSE↓ / Pearson / source-centred r / **pert-centred r** / top-20 방향 / DEG AUPRC / retrieval rank↓. 전체 표는 [loco_summary.tsv](F3/loco_summary.tsv).

| 보류 | 예측기 | RMSE | Pearson | src-c r | pert-c r | 방향 | AUPRC | retr. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| K562 (n=1,998, DEG 유병률 0.102) | no_change | 0.0668 | — | 0.460* | — | 0 | 0.102 | 0.500 |
| | global_mean | 0.0751 | 0.172 | — | — | 0.624 | 0.147 | 0.500 |
| | same_target_mean | 0.0821 | **0.302** | 0.358 | **0.335** | **0.791** | **0.253** | **0.089** |
| | nearest_line (Jurkat) | 0.0911 | 0.260 | 0.398 | 0.277 | 0.762 | 0.239 | 0.109 |
| | **calibrated_same_target** | **0.0651** | 0.291 | 0.481 | **0.335** | 0.781 | 0.224 | **0.089** |
| | ridge_pca | 0.0770 | 0.206 | −0.034 | 0.195 | 0.679 | 0.159 | 0.274 |
| RPE1 (n=2,200, 0.157) | no_change | 0.1160 | — | −0.095 | — | 0 | 0.157 | 0.500 |
| | global_mean | 0.1086 | 0.383 | — | — | 0.735 | 0.300 | 0.500 |
| | same_target_mean | 0.1053 | 0.343 | 0.296 | **0.355** | 0.780 | 0.304 | **0.117** |
| | nearest_line (HepG2) | 0.1177 | 0.299 | 0.263 | 0.304 | 0.761 | 0.271 | 0.166 |
| | **calibrated_same_target** | **0.1041** | **0.408** | 0.267 | **0.355** | **0.805** | **0.316** | **0.117** |
| | ridge_pca | 0.1068 | 0.380 | 0.108 | 0.193 | 0.758 | 0.293 | 0.330 |
| HepG2 (n=2,157, 0.087) | no_change | 0.0922 | — | 0.146 | — | 0 | 0.087 | 0.500 |
| | global_mean | 0.0895 | 0.281 | — | — | 0.702 | 0.165 | 0.500 |
| | same_target_mean | 0.0884 | 0.341 | 0.346 | **0.348** | 0.765 | **0.224** | **0.118** |
| | nearest_line (RPE1) | 0.1181 | 0.306 | 0.251 | 0.304 | 0.758 | 0.201 | 0.163 |
| | **calibrated_same_target** | **0.0827** | **0.354** | 0.372 | **0.348** | **0.776** | 0.221 | **0.118** |
| | ridge_pca | 0.0880 | 0.299 | 0.057 | 0.214 | 0.731 | 0.173 | 0.304 |
| Jurkat (n=2,264, 0.062) | no_change | 0.0847 | — | 0.322* | — | 0 | 0.062 | 0.500 |
| | global_mean | 0.0873 | 0.189 | — | — | 0.624 | 0.103 | 0.500 |
| | same_target_mean | 0.0895 | 0.293 | 0.338 | **0.305** | **0.769** | **0.194** | **0.096** |
| | nearest_line (K562) | 0.0922 | 0.257 | 0.376 | 0.269 | 0.745 | 0.172 | 0.112 |
| | **calibrated_same_target** | **0.0793** | 0.293 | 0.414 | **0.305** | 0.764 | 0.174 | **0.096** |
| | ridge_pca | 0.0885 | 0.211 | −0.020 | 0.153 | 0.668 | 0.111 | 0.313 |

\* **source-centred Pearson의 함정:** no-change의 source-centred 값은 corr(−m_src, y−m_src)다. 보류 세포주의 반응이 소스 평균 m_src보다 약하면 이 값은 크게 양수가 된다(K562 0.46). 즉 세포주 수준의 offset이 지표를 오염시킨다. 그래서 표적 특이 신호는 **pert-centred r과 retrieval**로 판단했다.

calibrated 대 no-change의 ΔRMSE 짝비교([headline_paired.tsv](F3/headline_paired.tsv)):

| 보류 | ΔRMSE [95% CI] |
|---|---|
| K562 | −0.00165 [−0.00225, −0.00102] |
| RPE1 | −0.01195 [−0.01250, −0.01140] |
| HepG2 | −0.00951 [−0.01010, −0.00893] |
| Jurkat | −0.00548 [−0.00603, −0.00492] |

나머지 짝비교 결과:

- **보정 없는 same_target 대 no-change:** K562 +0.0153, Jurkat +0.0048로 **악화**했고, RPE1 −0.0107, HepG2 −0.0037로 개선했다.
- **nearest-line 대 same_target 평균:** 4개 fold 모두에서 RMSE가 +0.003~+0.030 나빴고, pert-c r도 −0.04~−0.06 낮았다. 대조군 프로파일 유사도로 고른 단일 세포주는 여러 세포주 평균보다 열등하다.
- **ridge_pca 대 same_target:** pert-c r이 −0.13~−0.16 낮았다. ridge는 미관측 표적까지 예측할 수 있는 대신, 관측 표적의 특이 반응을 대부분 잃고 공통 반응에 가깝게 수렴한다.

**층화** ([headline_strata.tsv](F3/headline_strata.tsv), [loco_strata.tsv](F3/loco_strata.tsv)):

- calibrated의 RMSE 이득은 **효과 크기 상위 1/3**에 집중된다. 예: K562 0.1059→0.0917, HepG2 0.1436→0.1193.
- 하위 1/3에서는 오히려 no-change가 낫거나 같다. 예: K562 0.0363 vs 0.0440, Jurkat 0.0479 vs 0.0508.
- KD<40%인 표적(66–135개)에서는 같은 표적 전이의 AUPRC가 낮다. 예: Jurkat 0.064, 전체는 0.194.
- 주의: 효과가 약한 표적도 pert-c r이 0.24–0.36으로 나온다. 중심화 후에는 "평균보다 약한 반응"이라는 표적 수준의 크기 정보가 상관을 만들 수 있으므로, 유전자 특이 패턴과는 구분해야 한다.

### 4-2. 2×2 OOD 표 (같은 S_L 표적, 표적별 평균)

| 보류 | 관측 맥락·관측 표적 (D, 반쪽→반쪽) | 관측 맥락·미관측 표적 (C) | **미관측 맥락·관측 표적 (A on S_L)** | **미관측 맥락·미관측 표적 (Bs)** |
|---|---|---|---|---|
| K562 (n≈400) | pert-c r 0.362, AUPRC 0.478 | ridge RMSE **0.0615** (no-change 0.0645), pert-c r 0.194 | calibrated RMSE 0.0631 (0.0643), pert-c r **0.330**, AUPRC 0.218 | 최선 = **no-change** 0.0645; ridge 0.0759, pert-c r 0.160 |
| RPE1 (441) | 0.580, 0.607 | ridge **0.1015** (0.1174), 0.253 | calibrated 0.1057 (0.1174), **0.350**, 0.320 | ridge 0.1089 (0.1174), 0.161 |
| HepG2 (432) | 0.338, 0.333 | ridge 0.0851 (0.0897), 0.192 | calibrated **0.0811** (0.0897), **0.345**, 0.208 | ridge 0.0879 (0.0897), 0.123 |
| Jurkat (≈454) | 0.336, 0.405 | ridge 0.0792 (0.0831), 0.142 | calibrated **0.0781** (0.0830), **0.297**, 0.164 | 최선 = **no-change** 0.0831; ridge 0.0881, 0.114 |

출처: [ood_2x2.tsv](F3/ood_2x2.tsv), [headline_same_targets_A_vs_C.tsv](F3/headline_same_targets_A_vs_C.tsv).

- **D열 해석:** RMSE는 반쪽 정답의 잡음 때문에 no-change보다 커서 상한으로 쓸 수 없다. 반쪽 간 pert-c r을 신뢰도로 보고 Spearman-Brown으로 환산하면, 전체 세포 정답에 대한 상관 상한은 약 0.71(K562/HepG2/Jurkat)과 0.86(RPE1)이다. 두 반쪽이 같은 대조군 평균을 공유하므로 이 상한은 다소 과대추정이다.
- **교차 세포주가 세포주 내보다 얼마나 어려운가:**
  - 같은 S_L 표적에서 A(교차 세포주, 관측 표적)의 pert-c r은 C(세포주 내, 미관측 표적)보다 +0.10(RPE1)~+0.15(HepG2, Jurkat) 높다.
  - RMSE로는 HepG2 −0.0040, Jurkat −0.0011로 A가 낫고, K562 +0.0018, RPE1 +0.0042로 C가 낫다.
  - RPE1은 C의 Pearson(0.527)과 AUPRC(0.385)가 A(0.417, 0.320)보다 높다. 공통 반응이 강해 세포주 자체의 평균 반응이 유용하기 때문이다.
  - 표적을 소스에서 빼는 순간(Bs), 모든 지표가 C보다 나빠지고 no-change에 수렴한다(Bs ridge 대 no-change ΔRMSE: K562 +0.0114, Jurkat +0.0050, RPE1 −0.0085, HepG2 −0.0018).
  - 결론적으로 "미관측 세포주" 단독은 같은 표적 관측이 있으면 상당 부분 해결된다. **"미관측 표적 + 미관측 세포주"는 단순 기준선으로 사실상 풀리지 않는다.**
- **자연 이중 OOD(Bn, 3–23개 표적):** 너무 작아 결론을 내지 않는다. K562(23개)에서는 no-change RMSE 0.0864가 ridge 0.0912보다 낮았다.

## 5. 세포주별 실패 양상

- **K562:** 반응이 가장 약하다(DEG 중앙값 61, ‖Δ‖ 3.96). 소스 세포주의 크기를 그대로 옮기면 과대예측한다(same_target RMSE +0.015). 보정 후 이득은 −0.0017로 가장 작다. Bs에서는 no-change를 이기는 기준선이 없다. 라이브러리가 다르고(2,057 표적), 다른 세 세포주와 screen 설계가 다를 수 있다(day 미확인).
- **RPE1:** 반응이 가장 강하고 **공통 반응이 지배적**이다. global_mean만으로 Pearson 0.383, AUPRC 0.300으로 same_target 0.304와 비슷하다. 효과 상위 1/3에서는 global AUPRC(0.545)가 same_target(0.489)보다 높다. 표적 특이 전이의 추가 가치는 pert-c r(0.355)과 retrieval(0.117)에서만 분명하다. 공통 반응의 생물학적 원인(예: 스트레스/증식 반응)은 검증하지 않았다.
- **HepG2:**
  - 대조군이 가장 적고(4,976; gem_group 최소 5), 표적별 Welch DEG 수 중앙값이 HKU van Elteren의 약 4배(82 대 21)라서 DEG 지표의 신뢰도가 낮다.
  - nearest-line이 RPE1로 선택되면서 RMSE가 0.118로 크게 악화했다. RPE1의 큰 반응 크기를 그대로 옮긴 결과다.
  - calibrated 전이가 no-change 대비 −0.0095로 비교적 큰 이득을 보였다.
  - 간 유래 맥락의 특이 반응(예: 간 대사 유전자)을 따로 분석하지 않았다.
- **Jurkat:** K562와 유사하게 보정 없는 전이는 no-change보다 나쁘고(+0.0048), Bs에서는 no-change가 최선이다. KD<40%인 표적 135개는 전이 AUPRC가 0.064로, Jurkat 전체 DEG 유병률(0.062)과 비슷한 무작위 수준이다.

## 6. 한계

- **CRISPRi만 사용.** Norman K562 CRISPRa와 병합하거나 같은 지표로 비교하지 않았다. 화학적 개입(F4 SNU-423)과도 분리된다.
- **HepG2는 간모세포종 유래**이며 성인 HCC 근거가 아니다. GSE264667은 이전 V2 프로젝트에서 이미 본 자료로, 이번 결과는 개발용 재평가이지 **blind test가 아니다.** HepG2를 성인 HCC나 환자 반응 예측으로 일반화하지 않는다.
- **세포주 = 실험 묶음.** 세포주, 실험실(Replogle vs Nadig), 라이브러리, 시퀀싱 깊이, 수확 시점(파일에 없음)이 분리되지 않는다. LOCO는 이 묶음 전체를 넘는 전이를 측정한다.
- **배치·guide 교란:**
  - Δ는 gem_group 매칭 대조군을 쓰지만 DEG 검정의 대조군 분산은 세포주 전체 풀링이다(배치 층화 없음). HKU 비교에서 DEG 수는 검정 방식에 민감했다.
  - 다중 TSS 표적(118–135개)은 guide를 합쳐 표적 단위로 평균했다.
  - guide 효율 차이는 KD 층화로만 다뤘다.
- **정답 잡음:** 표적당 세포 수 중앙값이 45–121개이고 split-half pert-c r이 0.34–0.58이다. 따라서 모든 Pearson류 지표의 상한이 낮다.
- **통계:** bootstrap은 표적 단위이고, 표적은 경로·대조군을 공유하므로 독립이 아니다. 세포주 fold는 4개뿐이라 fold 간 일반화의 불확실성은 정량하지 않았다.
- **방법상 기록:**
  - v1 실행(job 20306)은 ridge 격자 K≤50, λ≤10을 썼고, 결과를 본 뒤 v2(job 20331)로 대체했다. 대체 이유는 (i) 내부 CV 선택이 격자 경계에 걸려 격자를 확장했고, (ii) 평가 전용 지표(pert-centred r, centroid 중심 retrieval)를 추가했기 때문이다.
  - 격자 확장은 보류 세포주 정답이 아니라 소스 내부 CV의 경계 선택이 계기였다. 선택 자체도 소스 내부 CV로만 했다.
  - same_target/calibrated/no-change/global/nearest의 RMSE·Pearson·AUPRC는 v1과 v2에서 동일하다. retrieval rank는 중심화 정의가 바뀌어 달라졌다. v1 표는 `assets/f3/runs/v1_job20306/`에 보존했다.
  - K=200은 여전히 경계지만 CV 곡면이 평평하다.
- **패널:** fold별로 다르다(소스 정의). 그래서 fold 간 절대 RMSE 비교는 제한적이다. 표적 유전자 2,035개를 포함하므로 고분산 유전자만 쓴 패널보다 반응 없는 유전자 비율이 높다.

## 7. 최신 모델(TxPert/State/PIE 등)이 넘어야 할 기준

같은 split과 패널을 써야 한다: [loco_test_splits.json](F3/loco_test_splits.json), `assets/f3/loco/panel_<L>.npy`, `assets/f3/matrices/<L>_delta.npz`. 선택은 소스 세포주 안에서만 한다.

| 조건 | 넘어야 할 기준 (fold별 K562 / RPE1 / HepG2 / Jurkat) |
|---|---|
| A 관측 표적·미관측 세포주 | RMSE < calibrated_same_target **0.0651 / 0.1041 / 0.0827 / 0.0793**; 동시에 pert-c r > same_target **0.335 / 0.355 / 0.348 / 0.305**, retrieval < **0.089 / 0.117 / 0.118 / 0.096**, DEG AUPRC > same_target **0.253 / 0.304 / 0.224 / 0.194** |
| Bs 미관측 표적·미관측 세포주 (S_L) | RMSE < 최선 기준(no-change **0.0645**, ridge **0.1089**, ridge **0.0879**, no-change **0.0831**); pert-c r > ridge **0.160 / 0.161 / 0.123 / 0.114**; retrieval < ridge **0.318 / 0.381 / 0.376 / 0.364** |
| C 세포주 내 미관측 표적 (S_L, 참조) | ridge RMSE 0.0615 / 0.1015 / 0.0851 / 0.0792 |

- RMSE 개선만으로는 부족하다. 크기 보정(shrinkage)만으로도 RMSE가 크게 바뀌기 때문이다. pert-centred r, retrieval, DEG AUPRC가 **같이** 개선돼야 표적 특이 반응을 맞힌 것으로 본다.
- 세포주 간 변이와 seed 변이를 구분하기 위해 4개 fold를 모두 보고해야 한다. HepG2는 사전 노출 자료로 표기해야 한다.
- PIE·State 등이 사전학습이나 prior 구축에 Replogle/Nadig 자료를 사용했다면, 보류 세포주의 처리 세포가 사전학습에 포함되었는지 먼저 확인해야 한다. 포함되었다면 LOCO 해석이 무효가 된다.

## 8. 재현·자원·산출물

- **환경:** 기존 `.venvs/scgpt`(Python 3.10, numpy 1.26.4, scipy 1.12.0, pandas 2.2.3, anndata 0.10.9, h5py 3.16.0, scikit-learn 1.5.2)를 읽기 전용으로 사용했고 설치는 하지 않았다. 동결 목록은 [env_freeze.txt](F3/env_freeze.txt).
- **코드:** `src/feedback/f3_pseudobulk.py`, `f3_contract.py`, `f3_loco.py`, `f3_crosscheck_hku.py`, `f3_report_tables.py`. 배치 스크립트는 `experiments/feedback/f3_pseudobulk.sbatch`, `f3_loco.sbatch`, `f3_crosscheck.sbatch`.
- **Slurm** ([slurm_jobs.tsv](F3/slurm_jobs.tsv)):

  | job | 내용 | 자원 | 결과 |
  |---|---|---|---|
  | 20277 (array 0–3) | pseudobulk | cpu, g4, 4 CPU/32G 각, 51–84 s | 완료 |
  | 20306 | LOCO v1 | c4, 16 CPU/96G, 9분 | 완료, 대체됨 |
  | **20331** | LOCO v2 (최종) | c4, 16 CPU/96G, 11분 | 완료 |
  | 20308–20310 | 교차검증 시도 | — | 실패: 인덱스명, 라벨 순서, 동시 덮어쓰기 |
  | **20333** | 교차검증 | c4, 4 CPU/48G, 1분 | 완료 |
  | 20303–20304 | 합성 자료 smoke test | — | — |
  | 20340 | 요약표 | — | — |

  sacct는 DB 연결 시간 초과로 실제 MaxRSS를 확보하지 못했다. 시간은 로그의 start/end를 기록했다. 20277은 `--exclude=g[1-4]`를 추가하기 전에 제출되어 g4(cpu 파티션 소속)에서 실행됐다.
- **G1 자산** (`assets/f3/`, git 제외): `pseudobulk/`(2.6 GB), `matrices/`(2.2 GB), `loco/`(예측 `pred_<L>_<cond>.npz`, 패널, `per_target_metrics.tsv.gz`, 444 MB), `runs/v1_job20306/`.
- **보고서 파일** (`reports/feedback/F3/`): `data_contract.{tsv,json}`, `target_overlap.tsv`, `line_pseudobulk_summary.tsv`, `cross_line_same_target_similarity.tsv`, `loco_folds.tsv`, `loco_hyperparams.json`, `loco_summary.tsv`, `loco_A_on_test_split.tsv`, `ood_2x2.tsv`, `paired_bootstrap.tsv`, `loco_strata.tsv`, `headline_*.tsv`, `loco_test_splits.json`, `loco_run_meta.json`, `crosscheck_hku*.{tsv,json}`, `env_freeze.txt`, `slurm_jobs.tsv`.

**상태:** 실측 완료(데이터 계약, pseudobulk, 4-fold LOCO 단순 기준선, 외부 QC). 미실행: 최신 모델, distributional distance, GO 기반 섭동 임베딩, 배치 층화 DEG 정답 민감도 분석. 차단: 없음.
