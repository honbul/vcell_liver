# G3 — R1 후보 PIE의 HepG2 LOCO 검증: 공식 체크포인트 재현과 단순 기준선 비교

**작성:** 2026-11-10 · 브랜치 `research/feedback-20261110`(커밋하지 않음) · 계획: [FEEDBACK_PLAN_20261110.md](../../docs/FEEDBACK_PLAN_20261110.md) G3 · 선행: [F2_LITERATURE_REVIEW.md](F2_LITERATURE_REVIEW.md) §4·§6(G1-a/G1-b), [F3_CROSS_CELL_LINE.md](F3_CROSS_CELL_LINE.md) · 실행: G1 Slurm(`cpu`, `gpu`).
**성격:** 공개 체크포인트 **평가와 재현**, 그리고 같은 split에서 우리가 만든 단순 기준선 비교다. PIE를 재학습하지 않았고, 하이퍼파라미터도 바꾸지 않았다. 아래 수치는 모두 실측이며, 계획이나 추정은 따로 표시했다.

---

## 0. 요약

1. **재현(G1-a): 12/12 근접 일치, 비트 단위 일치는 아님.**
   - 공식 체크포인트 4 fold × 3 조건을 1× A30에서 `pie eval`로 다시 돌렸다.
   - 모든 요약 지표가 HF 카드의 fold별 공개값과 |Δ| ≤ 0.0022 안에서 일치했다. 4-fold 평균 ± SD는 소수 셋째~넷째 자리까지 같다(예: DE AUPRC 0.4152/0.3048/0.2366, 공개 0.4155/0.3048/0.2366).
   - 쌍(pair) 집합은 동일하고, pair 단위 지표의 상관은 0.97–0.99999다. 차이는 bf16 연산과 GPU 기종 차이로 보인다(저자 H100, 우리 A30).
   - HepG2 fold에서는 CPU 추론도 기준 안에서 근접했다(|Δ| ≤ 0.0046). 다만 AUPRC가 0.002–0.005 체계적으로 낮았으므로, 공식 비교에는 GPU 결과를 썼다.
2. **같은 split, 같은 지표 코드(`pie.metrics.score`)로 본 PIE 대 기준선 (HepG2 fold, paired bootstrap 2,000회, perturbation 단위 재표집).** 조건별로 결과가 갈린다.

   | 조건 | PIE 공식 AUPRC | 최선 기준선 AUPRC | 판정 |
   |---|---:|---:|---|
   | **미관측 세포주·관측 표적** (`unseen_ctx`, n=867) | **0.442** | 0.391 (`same_target_cal`) | PIE 우세: +0.052 [0.045, 0.060]. 방향 +0.009, Spearman LFC +0.017, L1 판별 +0.255도 모두 유의 |
   | **미관측 perturbation** (`unseen_pert`, 소스 3개 세포주, 1,238쌍) | 0.321 | **0.414** (`context_mean_kd`) | 공식 AUPRC는 기준선 우세: −0.093 [−0.114, −0.072]. 방향(+0.029)·L1(+0.214)·Jaccard는 PIE 우세 |
   | **이중 OOD** (`unseen_ctx_pert`, n=386) | 0.244 | **0.381** (`global_mean_kd`) | 공식 AUPRC −0.136 [−0.167, −0.108], Spearman LFC −0.076은 기준선 우세. 방향은 차이 없음. L1·Jaccard는 PIE 우세 |

3. **해석 가능한 실패 ①: 표적 유전자 자기 자신.**
   - PIE 공식 AUPRC·방향·Spearman은 perturb된 유전자 자신을 포함한다. PIE 라벨에서 이 유전자는 학습 행의 96.6%에서 DE이고, log2FC 중앙값은 −28.4다.
   - PIE는 **학습에서 본 적 없는 perturbation**에서 표적 유전자의 `p_de`를 검정 유전자 중 **67–69 백분위**에 둔다. 관측 perturbation에서는 98 백분위다.
   - 그래서 약한 반응(DE 유전자가 적은) 행에서 AUPRC가 무너진다. 이중 OOD 하위 1/3에서 PIE는 0.025, 기준선은 0.420이다.
   - 기준선과 같은 "표적 유전자는 강하게 감소" 규칙을 PIE 예측에 사후로 덧씌우면(진단용, 공식값 아님) AUPRC는 0.441/0.387이 되어, 최선 기준선을 넘거나 비슷해진다.
4. **표적을 뺀 같은 정의의 지표로는 PIE가 우세하지만, 조건에 따라 크기가 다르다.**
   - AUPRC: unseen_ctx **+0.049** [0.040, 0.057], unseen_pert **+0.029** [0.021, 0.038], 이중 OOD **+0.007** [−0.001, 0.015, 유의하지 않음].
   - 방향 일치: +0.011 / +0.061 / +0.039.
   - perturbation 중심 상관(Δ): 미관측 perturbation에서 +0.19~+0.25. 이중 OOD에서도 ridge 대비 +0.065 [0.031, 0.097]이다.
   - 즉 **미관측 표적의 표적 특이 신호는 PIE에만 있다.**
5. **해석 가능한 실패 ②: Δ 크기 보정.**
   - PIE의 Δ(`delta_p_pred`) RMSE는 세 조건 모두 **no-change보다 나쁘다**(0.102 대 0.086, 0.092 대 0.081, 0.112 대 0.090).
   - 우리 원자료 정답(F3)으로 보면 PIE Δ 헤드의 상위 20 DEG 방향 일치는 0.63/0.58이다. 같은 모델의 LFC 헤드는 0.78/0.73이고, 기준선은 0.83/0.75다.
6. **F3 지표(우리 원자료 정답, HepG2 패널 3,141개 유전자)에서는 HepG2 이득이 사라진다.**
   - **관측 표적**에서는 정보량이 같은 기준선(`same_target_cal`)보다 RMSE(+0.036), Pearson(−0.030), DEG AUPRC(−0.026), 방향(−0.19) 모두 나쁘다. pert-centred r(−0.004)과 retrieval(+0.002)은 동등하다.
   - **이중 OOD**에서는 pert-centred r(+0.075)과 retrieval(−0.132, 낮을수록 좋음)만 ridge보다 낫다.
   - 두 전처리의 HepG2 정답은 표적별 Pearson 중앙값 0.986으로 거의 같다. 따라서 이 차이는 전처리 탓이 아니다.
7. **G3 판정: "해석 가능한 실패 증거 + 제한적 이득" → 조건부 통과.**
   - PIE는 미관측 perturbation의 표적 특이 패턴(순위·방향·판별)에서 단순 기준선이 못 하는 일을 한다.
   - 그러나 HepG2의 관측 표적 전이에서는 이득이 지표 정의에 따라 뒤집힌다.
   - 공식 AUPRC의 열세는 표적 유전자 처리 하나로 설명되고, Δ 크기는 보정되어 있지 않다.
   - G4로 넘길 때는 표적 제외 지표, LFC 헤드 방향, 사후 크기 보정을 함께 보고해야 한다(§5).
8. **보충(Jurkat·K562·RPE1 fold, 같은 코드): 같은 구조가 4/4 fold에서 반복된다.** 공식 AUPRC는 미관측 세포주에서 PIE 우세, 미관측 perturbation·이중 OOD에서 기준선 우세였다. 표적 p_de 백분위는 0.67–0.74, Δ RMSE는 no-change보다 나쁘다(12/12). HepG2 결과는 fold 고유 현상이 아니라 **모델 수준의 특성**이다(§9).

---

## 1. 설정

| 항목 | 값 |
|---|---|
| 코드 | `ArcInstitute/pie` **v1.0.1** = 커밋 `038940057545ae686db946ae247aeb0ff0101897`(2026-10-27). 체크포인트 카드 날짜(2026-10-05)는 v1.0.0(`b26dd17`)과 같은 날이다. 과제 지시대로 v1.0.1을 썼고, 재현이 근접했으므로 버전 차이 영향은 작다 |
| 환경 | `assets/envs/pie/` 전용. conda `python=3.12`(3.12.16) + `uv` 0.13.7로 `uv sync --frozen`(저장소 `uv.lock`). torch 2.10.0+cu128, numpy 2.4.6, pandas 3.0.3, lightning 2.6.5, huggingface-hub 1.21.0. 동결 목록: [G3/env_freeze_pie.txt](G3/env_freeze_pie.txt). 기존 `.venvs`에는 아무것도 설치하지 않았다. F3 지표 스크립트만 기존 `.venvs/scgpt`를 읽기 전용으로 사용했다 |
| 실행 설정 | `PIE_DATA_ROOT/RUNS_ROOT/CACHE_DIR` = `assets/g3_pie/{data,runs,cache}`, `WANDB_MODE=disabled`, HF 토큰 없음(모든 저장소 `gated=False`), 텔레메트리 끔 |
| GPU | g2 NVIDIA A30 24 GB × 1, 드라이버 565.57.01(CUDA 12.7). cu128 휠이 minor-version 호환으로 동작했다 |
| 라이선스 | 코드 **CC BY-NC-SA 4.0**. 체크포인트는 **Arc PIE Model Non-Commercial License + Acceptable Use Policy**. 데이터 저장소 3개(nadig_essential, sources, splits)에는 HF 라이선스 메타데이터가 없다 **[미확인]**. **학술·비상업 연구용으로만 사용**하고, 산업 협업 전에는 법무 확인이 필요하다 |

### 1-1. Provenance (모두 고정 커밋, [G3/provenance_fetch.json](G3/provenance_fetch.json))

| 자산 | HF 저장소 @ revision | 크기 | 비고 |
|---|---|---:|---|
| 체크포인트 4개 + config/data_stats/eval CSV | `arcinstitute/PIE_replogle_wdataset` @ `055c7a2cabdaff5659121c64fa2b609c231fc5be` | 4 × 949.6 MB | sha256: hepg2 `84826d8c3ee76be6…`, jurkat `e08f47cb2d7ccf49…`, k562 `b9d5a0b3943666d8…`, rpe1 `487dcab71aa16e30…`. **4/4가 HF LFS oid와 일치** |
| 전처리 자료 | `arcinstitute/pie_replogle_nadig_essential` @ `20c9faef76fc96fdc809871340bd93499413dfe7`/preprocessed | 0.76 GB | 체크포인트에 고정된 revision이다. 현재 main(`5119439…`)은 쓰지 않았다 |
| 지식원 7개 | `arcinstitute/pie_sources` @ `cb1aaa4e7655605bdc70a9bd77bbd62016b8c7d7` | 44.0 GB | esm2 0.10, **ncbi_text 43.21**, string_space 0.02, depmap_gene_effect 0.09, context_text <0.01, perturbation_text 0.17, gene_text 0.41 GB. 현재 main(`ae4ce640…`)은 쓰지 않았다 |
| split | `arcinstitute/PIE_splits` @ `396ab9563175ee887750c9eed7ccaea6f5fdbf50`/replogle_wdataset | <1 MB | 체크포인트 config의 `split_dir`도 같은 revision이다. 내려받은 `train.json`의 sha256(hepg2 `828efbc9…`)이 체크포인트에 고정된 `train_json_sha256`과 일치한다. evidence key(`f133ab23…`)도 일치했다. 불일치였다면 PIE가 오류를 낸다 |
| alias 표 | 코드 내 `src/pie/sources/curated_aliases.yaml`(체크포인트 config의 legacy 경로 매핑) | – | – |

**다운로드 총량:** 자산 44.76 GB + 체크포인트·CSV 3.6 GB ≈ **48.4 GB**로 예산 60 GB 안이다. 소요 시간은 575 s(xet 전송)였다. HF API로 크기를 먼저 확인했다(필요 없는 `jump_morphology`, `smiles` 등은 받지 않았다).

### 1-2. PIE의 HepG2 fold split 구조 (직접 확인)

- 데이터: 5,459 (세포주, perturbation) 행 × 6,642 유전자, 1,857 perturbation.
  - HepG2 행은 1,253개다(PIE 데이터 카드: 90,826 세포).
  - F3 원파일의 HepG2는 145,473 세포, ≥20 cells 표적 2,160개다. PIE는 knockdown 필터 등으로 세포와 표적이 적다.
- train: 2,670행(Jurkat 913, K562 834, RPE1 923). val: 298행.
  - val의 모든 perturbation은 다른 소스 세포주의 train에 있다. 즉 소스 세포주 안의 "미관측 맥락형" 홀드아웃이다.
  - 세 조건의 train/val은 동일하다(확인).
- test:
  - `unseen_ctx`: HepG2 867행. perturbation이 train에 1개 세포주(175), 2개(360), 3개(332)에서 관측된다.
  - `unseen_pert`: 소스 3개 세포주 1,238행(545 perturbation, train·val 어디에도 없음).
  - `unseen_ctx_pert`: HepG2 386행(train·val에 없음).
  - 867 + 386 = HepG2 1,253행 전부다. HepG2 처리 세포는 train/val 어디에도 없다(확인).
- 계획서 2×2와의 대응:
  - `unseen_ctx` = F3 조건 A.
  - `unseen_ctx_pert` = 이중 OOD(F3의 Bs와 표적 집합이 다르다).
  - `unseen_pert` = **소스 세포주 안**의 미관측 표적(F3 C와 비슷하지만 HepG2가 아니다).

---

## 2. G1-a: 공식 체크포인트 재현 대 공개값

명령은 HF 카드 그대로다(`pie eval experiment_name=replogle_wdataset/<fold> ckpt=best_auprc split_path=hf://…@396ab95…/<setting>/<fold>/test.json row_set=<setting>`). `save_predictions=true`를 추가했고, `run_dir`은 우리 디렉터리로 분리했다. 일치 기준은 다음과 같다: 모든 요약 지표가 |Δ| ≤ 1e-6이면 exact, ≤ 0.005이면 close, 그 밖은 mismatch.

| fold | 조건 | 상태 | 최대 \|Δ\| | AUPRC 재현 / 공개 | 방향 재현 / 공개 |
|---|---|---|---:|---|---|
| **hepg2** | unseen_ctx | close | 0.0012 | 0.4421 / 0.4432 | 0.8403 / 0.8401 |
| **hepg2** | unseen_pert | close | 0.0007 | 0.3214 / 0.3214 | 0.7822 / 0.7819 |
| **hepg2** | unseen_ctx_pert | close | 0.0004 | 0.2444 / 0.2444 | 0.7200 / 0.7196 |
| jurkat | ctx / pert / ctx_pert | close ×3 | ≤ 0.0022 | 0.3755 / 0.3257 / 0.2139 (공개 0.3754 / 0.3255 / 0.2140) | – |
| k562 | ctx / pert / ctx_pert | close ×3 | ≤ 0.0008 | 0.4222 / 0.3028 / 0.1984 (공개 0.4222 / 0.3028 / 0.1984) | – |
| rpe1 | ctx / pert / ctx_pert | close ×3 | ≤ 0.0004 | 0.4211 / 0.2694 / 0.2897 (공개 0.4211 / 0.2694 / 0.2897) | – |
| hepg2 (**CPU** 추론) | unseen_ctx / unseen_pert / unseen_ctx_pert | close ×3 | 0.0046 / 0.0027 / 0.0026 | 0.4387 / 0.3187 / 0.2425 (공개 0.4432 / 0.3214 / 0.2444) | 0.8398 / 0.7824 / 0.7170 |

4-fold 평균 ± SD (ddof=1). 위가 재현, 아래가 HF 카드다.

| 조건 | AUPRC | Jaccard | 방향 | Spearman DE 수 | Spearman LFC | L1 판별 |
|---|---|---|---|---|---|---|
| unseen_ctx | 0.4152 ± 0.0282 | 0.1773 ± 0.0126 | 0.8236 ± 0.0205 | 0.7389 ± 0.0394 | 0.6567 ± 0.0409 | 0.7700 ± 0.0604 |
| (공개) | 0.4155 ± 0.0286 | 0.1771 ± 0.0128 | 0.8235 ± 0.0206 | 0.7389 ± 0.0395 | 0.6568 ± 0.0411 | 0.7701 ± 0.0604 |
| unseen_pert | 0.3048 ± 0.0256 | 0.1671 ± 0.0115 | 0.7776 ± 0.0135 | 0.5420 ± 0.0350 | 0.5502 ± 0.0199 | 0.7143 ± 0.0106 |
| (공개) | 0.3048 ± 0.0256 | 0.1670 ± 0.0114 | 0.7775 ± 0.0135 | 0.5420 ± 0.0350 | 0.5505 ± 0.0201 | 0.7142 ± 0.0106 |
| unseen_ctx_pert | 0.2366 ± 0.0402 | 0.1160 ± 0.0198 | 0.7251 ± 0.0188 | 0.4657 ± 0.0719 | 0.4473 ± 0.0472 | 0.6527 ± 0.0429 |
| (공개) | 0.2366 ± 0.0402 | 0.1160 ± 0.0198 | 0.7257 ± 0.0197 | 0.4658 ± 0.0717 | 0.4475 ± 0.0469 | 0.6526 ± 0.0429 |

- **판정: 재현 성공(근접 일치), bit-exact는 아님.** 출처: [repro_vs_published.tsv](G3/repro_vs_published.tsv), [_4fold.tsv](G3/repro_vs_published_4fold.tsv), [_pairs.tsv](G3/repro_vs_published_pairs.tsv).
- HepG2 3개 조건에서 공개 per-pair 값과 재현값의 paired bootstrap 차이는 모두 |Δ| ≤ 0.0012이고, 95% CI가 대부분 0을 포함한다.
- 개별 pair의 큰 차이(방향 최대 1.0)는 DE 유전자가 1–2개인 행에서 거의 동점인 예측의 부호가 뒤집힌 경우다.

---

## 3. G1-b: PIE split에서의 단순 기준선 (PIE 지표 그대로)

### 3-1. 정의와 선택 규칙

코드는 [g3_pie_baselines.py](../../src/feedback/g3_pie_baselines.py)이다.

- **입력:** PIE 전처리 자료(유전자 축 6,642개, DE 정의는 Mann-Whitney + BH < 0.05 & `tested`).
- **채점:** 모든 기준선은 `p_de`, `lfc_pred`, `delta_p_pred`를 출력하고, **`pie.metrics.score`가 그대로 채점**한다. 공개 수치와 정의가 같다.
- **가시 자료:** 소스 3개 세포주의 train/val 행과 각 세포주의 control 평균뿐이다.
- **선택 규칙:** 선택·적합은 소스 세포주 안에서만 했다. 테스트 행은 채점에만 썼다.

| 예측기 | 내용 | 선택 |
|---|---|---|
| `no_change` | p_de=0, lfc=0, Δ=0 | – |
| `*_kd` | **표적 유전자 규칙**: 표적의 lfc = train 표적 LFC 중앙값(−28.4), Δ = (Δ/ctrl 중앙값 −1.0) × ctrl, p_de ≥ train 표적 DE 비율(0.966) | train |
| `global_mean(_kd)` | 소스 세포주별 train 평균(LFC, Δ)의 등가중 평균. p_de = 유전자별 DE 빈도 | – |
| `context_mean(_kd)` | (unseen_pert) 시험 세포주 자신의 train 평균 | – |
| `nearest_line_mean_kd` | control 평균 상관이 가장 높은 소스(RPE1 0.913 > K562 0.906 > Jurkat 0.889)의 평균 반응 | control만 사용 |
| `same_target_raw` | (unseen_ctx) 같은 perturbation의 소스 train 행 평균. p_de = 평균(1−FDR) | – |
| `same_target_cal` | lfc = 0.076·same + 0.236·global, Δ = 0.347·same + 0.245·global. p_de = 로지스틱(동일 표적 DE 비율·1−FDR·\|lfc\|·\|Δ\|·global DE 빈도) | **val 298행**(최소제곱·로지스틱) |
| `nearest_line_same_target` | 가장 가까운 소스 세포주의 같은 perturbation 행 | – |
| `ridge_stacked_kd` / `ridge_inline_kd` | Ahlmann-Eltze식 쌍선형 ridge(F3와 같은 추정식). 유전자 임베딩 = train LFC의 상위 K PC. stacked(소스 3개)와 inline(시험 세포주 자신) 두 가지 | **perturbation 단위 5-fold CV**(train). 선택값 K=5, λ=10⁴(stacked). K562 inline만 K=200. λ가 격자 경계라서 ridge가 사실상 평균으로 수축 |

**진단용(공식 결과 아님):** `PIE_repro_kd` = 재현한 PIE 예측에 같은 표적 규칙을 덧씌운 것.

**추가 지표:** 공식 6개 외에, 같은 정의에서 **표적 유전자를 뺀** AUPRC_x·방향_x·Spearman_x를 계산했다. 또 표적 유전자의 `p_de` 백분위, Δ 척도의 RMSE·Pearson·perturbation 중심 Pearson(조건 내 세포주별 centroid 차감)도 함께 봤다.

### 3-2. 결과 (HepG2 fold) — [pie_split_summary_hepg2.tsv](G3/pie_split_summary_hepg2.tsv)

열은 공식 지표(AUPRC / Jaccard / 방향 / Sp.LFC / L1)와 표적 제외 지표(AUPRC_x / 방향_x), 표적 p_de 백분위, Δ 척도 지표(RMSE↓ / pert-c r)다.

**(a) 미관측 세포주·관측 표적 `unseen_ctx` (HepG2, n=867)**

| 예측기 | AUPRC | Jacc. | 방향 | Sp.LFC | L1 | AUPRC_x | 방향_x | 표적 pct | Δ RMSE | pert-c r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_change | 0.078 | 0.110 | 0 | – | 0.502 | 0.078 | 0 | 0.50 | 0.0865 | – |
| global_mean_kd | 0.363 | 0.004 | 0.733 | 0.564 | 0.502 | 0.220 | 0.634 | 1.00 | 0.0830 | 0.01 |
| same_target_raw | 0.369 | 0.127 | 0.824 | 0.645 | 0.736 | 0.267 | 0.733 | 0.99 | 0.0886 | 0.305 |
| nearest_line_same_target | 0.291 | 0.098 | 0.806 | 0.624 | 0.745 | 0.231 | 0.716 | 0.98 | 0.1052 | 0.260 |
| **same_target_cal** | 0.391 | 0.151 | 0.831 | 0.694 | 0.558 | 0.282 | 0.741 | 1.00 | **0.0787** | 0.305 |
| ridge_stacked_kd | 0.363 | 0.025 | 0.740 | 0.569 | 0.503 | 0.220 | 0.640 | 1.00 | 0.0829 | 0.118 |
| **PIE (재현)** | **0.442** | **0.162** | **0.840** | **0.711** | **0.813** | **0.331** | **0.752** | 0.98 | 0.1016 | **0.311** |
| PIE (공개 CSV) | 0.443 | 0.162 | 0.840 | 0.711 | 0.813 | – | – | – | – | – |

**(b) 미관측 perturbation `unseen_pert` (Jurkat/K562/RPE1, 1,238쌍, 545 perturbation)**

| 예측기 | AUPRC | Jacc. | 방향 | Sp.LFC | L1 | AUPRC_x | 방향_x | 표적 pct | Δ RMSE | pert-c r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_change | 0.092 | 0.077 | 0 | – | 0.502 | 0.092 | 0 | 0.50 | 0.0806 | – |
| no_change_kd | 0.209 | 0.077 | 0.118 | 0.241 | 0.502 | 0.092 | 0 | 1.00 | 0.0806 | – |
| context_mean (규칙 없음) | 0.293 | 0.072 | 0.709 | 0.404 | 0.503 | 0.295 | 0.665 | 0.59 | 0.0756 | – |
| **context_mean_kd** | **0.414** | 0.072 | 0.753 | 0.529 | 0.503 | 0.295 | 0.665 | 1.00 | 0.0756 | 0.02 |
| ridge_inline_kd | 0.413 | 0.071 | 0.757 | 0.530 | 0.503 | 0.293 | 0.669 | 1.00 | **0.0755** | 0.080 |
| ridge_stacked_kd | 0.413 | 0.061 | 0.729 | 0.475 | 0.503 | 0.293 | 0.641 | 1.00 | 0.0781 | 0.111 |
| **PIE (재현)** | 0.321 | **0.171** | **0.782** | **0.539** | **0.717** | **0.324** | **0.725** | **0.69** | 0.0919 | **0.269** |
| PIE + 표적 규칙(진단) | 0.441 | 0.171 | 0.809 | 0.618 | 0.717 | 0.324 | 0.725 | 1.00 | 0.0919 | 0.269 |

**(c) 이중 OOD `unseen_ctx_pert` (HepG2, n=386)**

| 예측기 | AUPRC | Jacc. | 방향 | Sp.LFC | L1 | AUPRC_x | 방향_x | 표적 pct | Δ RMSE | pert-c r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_change | 0.097 | 0.119 | 0 | – | 0.502 | 0.097 | 0 | 0.50 | 0.0902 | – |
| no_change_kd | 0.238 | 0.119 | 0.141 | 0.265 | 0.502 | 0.097 | 0 | 1.00 | 0.0902 | – |
| global_mean (규칙 없음) | 0.237 | 0.003 | 0.684 | 0.449 | 0.502 | 0.238 | 0.623 | 0.61 | 0.0867 | – |
| **global_mean_kd** | **0.381** | 0.003 | 0.727 | 0.577 | 0.502 | 0.238 | 0.623 | 1.00 | 0.0867 | 0.02 |
| nearest_line_mean_kd | 0.376 | 0.046 | 0.724 | 0.582 | 0.503 | 0.234 | 0.620 | 1.00 | 0.0937 | 0.02 |
| ridge_stacked_kd | 0.382 | 0.023 | 0.727 | 0.578 | 0.503 | 0.240 | 0.623 | 1.00 | **0.0865** | 0.144 |
| **PIE (재현)** | 0.244 | **0.141** | 0.720 | 0.501 | **0.692** | **0.245** | **0.663** | **0.67** | 0.1120 | **0.209** |
| PIE + 표적 규칙(진단) | 0.387 | 0.141 | 0.759 | 0.597 | 0.692 | 0.245 | 0.663 | 1.00 | 0.1120 | 0.209 |

### 3-3. Paired bootstrap (PIE − 기준선, 95% CI)

표적 단위 재표집 2,000회다. 통계량은 PIE와 같은 집계(세포주별 pair 평균 → 세포주 평균)다. [pie_split_paired_bootstrap_hepg2.tsv](G3/pie_split_paired_bootstrap_hepg2.tsv), [_kd_diag_hepg2.tsv](G3/pie_split_paired_bootstrap_kd_diag_hepg2.tsv).

| 조건 (기준선) | AUPRC (공식) | 방향 (공식) | Sp.LFC (공식) | L1 | **AUPRC_x** | **방향_x** | **Sp.LFC_x** | **pert-c r (Δ)** | Δ RMSE (↓) |
|---|---|---|---|---|---|---|---|---|---|
| unseen_ctx (same_target_cal) | **+0.052** [0.045, 0.060] | +0.009 [0.001, 0.016] | +0.017 [0.003, 0.032] | **+0.255** | **+0.049** [0.040, 0.057] | +0.011 [0.002, 0.020] | +0.017 [−0.006, 0.040] | +0.006 [−0.001, 0.013] | +0.023 [0.022, 0.024] **나쁨** |
| unseen_pert (context_mean_kd) | **−0.093** [−0.114, −0.072] | +0.029 [0.013, 0.044] | +0.010 [−0.018, 0.037] | **+0.214** | **+0.029** [0.021, 0.038] | **+0.061** [0.048, 0.074] | **+0.102** [0.076, 0.126] | **+0.249** [0.226, 0.271] | +0.016 [0.014, 0.018] **나쁨** |
| unseen_pert (ridge_inline_kd) | −0.092 [−0.113, −0.071] | +0.025 [0.010, 0.039] | +0.009 [−0.019, 0.035] | +0.214 | +0.031 [0.022, 0.039] | +0.056 [0.043, 0.069] | +0.097 [0.072, 0.123] | **+0.189** [0.167, 0.210] | +0.016 **나쁨** |
| unseen_ctx_pert (global_mean_kd) | **−0.136** [−0.167, −0.108] | −0.008 [−0.033, 0.014] | **−0.076** [−0.122, −0.031] | **+0.189** | +0.007 [−0.001, 0.015] | **+0.039** [0.018, 0.061] | +0.047 [0.006, 0.093] | +0.192 | +0.025 [0.023, 0.028] **나쁨** |
| unseen_ctx_pert (ridge_stacked_kd) | −0.138 [−0.168, −0.110] | −0.007 [−0.032, 0.018] | −0.077 [−0.123, −0.035] | +0.188 | +0.006 [−0.003, 0.014] | +0.040 [0.016, 0.062] | +0.044 [0.003, 0.087] | **+0.065** [0.031, 0.097] | +0.026 **나쁨** |

- **진단: PIE + 표적 규칙 대 최선 기준선(공식 AUPRC):** unseen_ctx +0.061 [0.053, 0.070], unseen_pert +0.027 [0.019, 0.035], 이중 OOD +0.006 [−0.001, 0.014].
- 규칙 없는 평균 기준선과 비교하면 PIE 원본이 공식 AUPRC에서 앞서거나 비슷하다. unseen_pert에서 context_mean 대비 +0.028 [0.020, 0.036], 이중 OOD에서 global_mean 대비 +0.007 [−0.000, 0.015]다.
- 즉 **공식 AUPRC의 역전은 거의 전부 표적 유전자 1개의 순위**에서 나온다.

### 3-4. DE 수 3분위 층화 ([pie_split_strata_hepg2.tsv](G3/pie_split_strata_hepg2.tsv))

3분위 경계(DE 수)는 unseen_ctx 12/442, unseen_pert 38/536, 이중 OOD 13/662다. 각 칸은 공식 AUPRC (표적 제외 AUPRC_x)이고, 기준선은 각 조건의 최선(same_target_cal / context_mean_kd / global_mean_kd)이다.

| 조건 | 하위 1/3 PIE | 하위 1/3 기준선 | 중간 PIE | 중간 기준선 | 상위 1/3 PIE | 상위 1/3 기준선 |
|---|---|---|---|---|---|---|
| unseen_ctx | 0.524 (0.210) | 0.486 (0.179) | 0.279 (0.263) | 0.242 (0.226) | 0.520 (0.519) | 0.443 (0.442) |
| unseen_pert | **0.089** (0.096) | 0.375 (0.037) | 0.272 (0.273) | 0.282 (0.271) | 0.609 (0.609) | 0.584 (0.583) |
| 이중 OOD | **0.025** (0.027) | 0.420 (0.015) | 0.160 (0.161) | 0.190 (0.169) | 0.547 (0.547) | 0.531 (0.530) |

- 약한 반응 행에서는 DE 유전자가 몇 개뿐이고 그중 하나가 표적이다. 그래서 표적 순위가 AUPRC를 결정한다.
- 표적을 빼면 PIE는 이중 OOD 중간 층(0.161 대 0.169)을 제외한 모든 층에서 기준선과 같거나 높다. 다만 이중 OOD 상위 1/3의 우위도 +0.017에 그친다.

---

## 4. F3 지표로 본 PIE의 HepG2 예측 (우리 원자료 정답)

코드는 [g3_pie_f3metrics.py](../../src/feedback/g3_pie_f3metrics.py)이고, F3의 `per_target_metrics`·`retrieval_rank`를 그대로 썼다.

- **평가 단위:** PIE test 표적(unseen_ctx 867, 이중 OOD 386)은 **모두 F3 HepG2 표적에 있다.** 유전자는 F3 HepG2 fold 패널 3,141개로, 전부 PIE 축에 있다.
- **전처리 차이:**

  | 항목 | PIE | F3 |
  |---|---|---|
  | 세포 | knockdown 필터 후 90,826 세포 | 원파일 145,473 세포 전부 |
  | 정답 Δ | 세포주 pooled control 대비 pseudobulk log1p 차이 | gem_group 구성비로 매칭한 control 대비 평균 log1p(CP10k, `UMI_count`) 차이 |
  | DE | Mann-Whitney/BH, ≥5 CPM | Welch t/BH, 6,642 유전자 |

- **정답 일치도:** PIE 라벨을 그대로 예측으로 넣으면 F3 정답과 표적별 Pearson 중앙값 **0.986**, RMSE 0.020이다. PIE Δ의 노름이 1.11배 크다(knockdown 필터 효과로 추정). DEG 정의 차이 때문에 DEG AUPRC 상한은 0.54–0.57이다.
- **정보량 주의:** `F3_*` 행은 F3 원자료 기준선이다. F3는 PIE split과 무관하게 소스의 모든 (세포주, 표적)을 썼으므로 PIE val/unseen_pert 쌍도 포함한다. 그래서 정보량이 같은 비교는 PIE split 기준선(`same_target_cal` 등)이다.

열: RMSE↓ / Pearson / pert-centred r / top-20 DEG 방향 / DEG AUPRC / retrieval rank↓ ([f3metrics_summary_hepg2.tsv](G3/f3metrics_summary_hepg2.tsv)).

| 조건 | 예측기 | RMSE | Pearson | pert-c r | 방향 | AUPRC | retr. |
|---|---|---:|---:|---:|---:|---:|---:|
| unseen_ctx (867) | no_change | 0.0951 | – | – | 0 | 0.113 | 0.500 |
| | global_mean_kd | 0.0909 | 0.328 | 0.01 | 0.733 | 0.211 | 0.500 |
| | **same_target_cal** (PIE split) | **0.0838** | **0.399** | **0.355** | **0.827** | **0.261** | **0.123** |
| | ridge_stacked_kd | 0.0908 | 0.331 | 0.129 | 0.736 | 0.212 | 0.424 |
| | **PIE Δ 헤드** | 0.1201 | 0.369 | 0.352 | 0.634 | 0.234 | 0.125 |
| | PIE LFC 헤드 (척도 무관 지표만) | – | 0.146 | 0.100 | 0.782 | 0.154 | 0.170 |
| | F3_calibrated_same_target (참조) | 0.0830 | 0.421 | 0.389 | 0.831 | 0.277 | 0.093 |
| | PIE 라벨 자체 (전처리 상한) | 0.0214 | 0.980 | 0.982 | 1.000 | 0.542 | 0.000 |
| 이중 OOD (386) | no_change | 0.0987 | – | – | 0 | 0.133 | 0.500 |
| | global_mean_kd | 0.0945 | 0.329 | 0.02 | 0.744 | 0.238 | 0.500 |
| | **ridge_stacked_kd** | **0.0943** | **0.332** | 0.156 | **0.745** | **0.240** | 0.388 |
| | **PIE Δ 헤드** | 0.1324 | 0.292 | **0.232** | 0.576 | 0.213 | **0.255** |
| | PIE LFC 헤드 | – | 0.094 | 0.049 | 0.727 | 0.153 | 0.302 |

**Paired bootstrap (PIE Δ − 기준선, 표적 2,000회, [f3metrics_paired_bootstrap_hepg2.tsv](G3/f3metrics_paired_bootstrap_hepg2.tsv)):**

- **unseen_ctx 대 same_target_cal:**

  | 지표 | 차이 [95% CI] | 해석 |
  |---|---|---|
  | RMSE | +0.036 [0.034, 0.038] | PIE 나쁨 |
  | Pearson | −0.030 [−0.036, −0.024] | PIE 나쁨 |
  | pert-c r | −0.004 [−0.011, 0.004] | 차이 없음 |
  | 방향 | −0.193 [−0.206, −0.179] | PIE 나쁨 |
  | AUPRC | −0.026 [−0.031, −0.022] | PIE 나쁨 |
  | retrieval | +0.002 [−0.006, 0.010] | 차이 없음 |

  F3_calibrated(원자료 기준선) 대비로는 모든 지표에서 PIE가 나쁘다.
- **이중 OOD 대 ridge_stacked_kd:**

  | 지표 | 차이 [95% CI] | 해석 |
  |---|---|---|
  | RMSE | +0.038 [0.035, 0.042] | PIE 나쁨 |
  | Pearson | −0.040 [−0.056, −0.025] | PIE 나쁨 |
  | **pert-c r** | **+0.075** [0.041, 0.109] | PIE 우세 |
  | 방향 | −0.170 [−0.194, −0.146] | PIE 나쁨 |
  | AUPRC | −0.027 [−0.035, −0.020] | PIE 나쁨 |
  | **retrieval** | **−0.132** [−0.166, −0.098] | PIE 우세(낮을수록 좋음) |

- **해석:**
  - PIE는 표적별 패턴(무엇이 무엇과 다른가)을 기준선보다 잘 구분한다.
  - 그러나 Δ의 **크기와 부호**는 기준선보다 나쁘다.
  - LFC 헤드는 부호(0.78)가 Δ 헤드(0.63)보다 훨씬 낫지만 Δ 척도 상관은 낮다. 두 출력 헤드가 서로 일관되지 않는다.
  - F3 HepG2 조건 A 전체(2,157 표적)의 F3 보고값(calibrated RMSE 0.0827)과 이 867개 부분집합 값(0.0830)은 비슷하다.

---

## 5. G3 Gate 판정

**Gate 문구:** "반응 관련 지표에서 baseline 대비 이득 또는 해석 가능한 실패 증거"

**판정: 조건부 통과(이득은 제한적, 실패는 해석 가능).**

| OOD 조건 | 이득 증거 | 실패·한계 증거 | 한 줄 결론 |
|---|---|---|---|
| 관측 표적·미관측 세포주 (HepG2) | PIE 지표·표적 제외 AUPRC에서 +0.05, L1 판별 +0.26 | F3 정답 기준으로는 같은 표적 평균+보정 기준선에 RMSE·Pearson·방향·AUPRC 모두 열세. pert-c r·retrieval 동등. Δ RMSE는 no-change보다도 나쁨 | **같은 표적 전이를 넘어서는 신뢰할 만한 HepG2 이득은 없다** (지표·라벨 정의에 따라 부호가 바뀜) |
| 미관측 perturbation (소스 세포주) | 표적 제외 AUPRC +0.03, 방향 +0.06, Spearman +0.10, pert-c r +0.19~0.25 — 기준선이 원리적으로 못 하는 표적 특이 신호 | 공식 AUPRC −0.09(표적 유전자 미예측), Δ 크기 미보정 | **실질 이득**(표적 특이 패턴). 단 표적 유전자 처리와 크기 보정이 필요 |
| 이중 OOD (HepG2) | 표적 제외 방향 +0.04, pert-c r +0.07(PIE 지표·F3 지표 모두), retrieval 우세 | 공식 AUPRC −0.14, Spearman −0.08, 표적 제외 AUPRC는 평균 사전과 동등(+0.007, n.s.). F3 방향·AUPRC·RMSE 열세 | **약한 표적 특이 신호만** 있다. DE 집합 복원은 "자주 DE되는 유전자" 사전 수준 |

**해석 가능한 실패 세 가지:**

1. **표적 유전자 미예측.** 미관측 perturbation에서 PIE는 자신이 perturb한 유전자를 DE로 강하게 예측하지 않는다(67–69 백분위). PIE 학습은 표적 유전자의 LFC Huber 손실에 α=0.001 가중을 둔다(`lfc_target_gene_alpha`). DE 손실에는 그런 가중이 없다. 원인은 모델 설계 또는 학습 자료로 보이지만 확인하지 않았다 **[가설]**. 공식 지표가 표적을 포함하므로, 이 한 가지가 약한 반응 행의 AUPRC를 지배한다.
2. **Δ 크기 미보정.** RMSE는 no-change보다 나쁘고, F3 정답 기준 상위 DEG 방향은 0.58–0.63으로 같은 모델의 LFC 헤드(0.73–0.78)보다 낮다.
3. **공통 반응 사전이 강하다.** 유전자별 DE 빈도만으로 표적 제외 AUPRC 0.22–0.29를 얻는다. PIE는 그 위에 +0.007~+0.049를 얹는 정도다.

**G4 진행 시 필수 조건(계획):**

- (i) 공식 지표와 **표적 제외 지표를 함께** 보고한다.
- (ii) 소스 val에서 맞춘 **사후 크기 보정**(PIE Δ에 a·Δ+b)을 같은 split에서 시험한다. 이것은 G4 후보 수정이며, 이번 G3 결과로 선택하지 않았다.
- (iii) 방향 지표는 LFC 헤드로 본다.
- (iv) TxPert/State 같은 다른 후보도 같은 기준선 묶음과 표적 제외 지표로 비교한다.
- (v) 4 fold 전체에서 반복한다. 다른 3 fold의 기준선 결과는 §9의 보충표에 있다.

---

## 6. HepG2 사전 노출과 생물학적 범위

- **우리 쪽 노출.** GSE264667(HepG2)은 이전 V2 프로젝트에서 이미 본 자료다(`manifests/exposure_ledger.tsv`: development_previously_exposed). 이번 평가는 **blind test가 아니다.** PIE 저자도 HepG2 fold의 test 결과를 공개했으므로, 방법 수준의 설계 선택이 HepG2 결과를 보고 이루어졌을 가능성을 배제할 수 없다 **[미확인]**.
- **PIE의 HepG2 fold가 본 것.**
  - HepG2 처리 세포: train/val 어디에도 없다(확인).
  - HepG2 control 평균: 입력으로 사용한다(계획서의 "대상 세포주 control만 관측" regime과 같다).
  - context_text(Cellosaurus 기반): HepG2 설명에 "Disease: Hepatoblastoma, Age 15 years, NRAS p.Q61L, TERT promoter C228T"가 들어 있다(직접 확인).
  - **DepMap CRISPRGeneEffect** 지식원: 1,186개 세포주 열 가운데 **ACH-000739**가 포함된다(직접 확인). 이것이 HepG2의 DepMap ID로 알려져 있으나, DepMap 메타데이터로 다시 대조하지는 않았다. 따라서 모든 perturbation 임베딩에 HepG2 고유의 CRISPR knock-out 적합도 정보가 들어 있다. Perturb-seq 발현은 아니지만, "미관측 맥락"이 사전지식 면에서 완전히 미관측은 아니다.
  - ncbi_text/gene_text 문헌 텍스트에 Nadig 2025 결과가 반영되었는지는 확인하지 않았다 **[미확인]**.
- **생물학적 범위.** HepG2는 **15세 남성의 간모세포종(hepatoblastoma) 유래**다. 성인 HCC나 환자 반응의 근거가 아니며, 이번 결과를 HCC 예측 성능으로 일반화하지 않는다.
- **세포주 = 실험 묶음.** Replogle(K562/RPE1)과 Nadig(HepG2/Jurkat)의 실험실·라이브러리·깊이가 세포주와 분리되지 않는다(F3와 같은 한계).

---

## 7. 자원 사용 (실측) — [G3/slurm_jobs.tsv](G3/slurm_jobs.tsv)

| 작업 | job | 노드 / 자원 | 시간 |
|---|---|---|---|
| 자산 다운로드 | 20353 | c4, 4 CPU / 24G | 575 s |
| **공식 평가 12회 (GPU)** | **20403** | g2, **A30 × 1**, 8 CPU / 96G | 약 12분(평가당 32–92 s) |
| 공식 평가 HepG2 (CPU, 장치 민감도) | 20400 | c5, 32 CPU / 128G | ctx_pert 666 s, ctx 1,347 s, pert 1,860 s(GPU 대비 약 16–30배) |
| 기준선 + bootstrap (HepG2, 최종) | 20449 | c4, 16 CPU / 96G | 969 s(적합 140 s) |
| F3 지표 (최종) | 20472 | c4, 8 CPU / 48G | 약 1분 |
| 보충: 기준선 3 fold (bootstrap 500) | 20478/20480/20482 → F3 지표 20479/20481/20483 | c3/c4, 16 CPU / 96G (F3 8 CPU) | 기준선 각 약 15분, F3 각 약 1분 |

- **실패·취소:**
  - 20355: 체크포인트 다운로드 전에 GPU가 할당되어 "checkpoint not found"로 실패했다(제출 순서 실수). 20403으로 재제출했다.
  - 20358/20361/20450: 스크립트 버그로 실패했고, 수정했다.
  - 20440–20442, 20447/20448: GPU가 다른 fold를 처리해 불필요해져서 취소했다.
  - 다른 사용자의 작업은 건드리지 않았다.
- **측정 공백:** GPU 최대 메모리는 따로 측정하지 못했다. `nvidia-smi` 질의가 노드 전체 GPU를 보여 줬고, `sacct` DB 연결은 시간 초과였다. HF 카드 기준 평가는 1 GPU면 된다.
- **디스크 (`assets/`):**

  | 위치 | 크기 |
  |---|---:|
  | `g3_pie/data` | 42 GB |
  | `g3_pie/models` | 3.6 GB |
  | `g3_pie/runs` | 0.3 GB(예측 parquet 포함) |
  | `g3_pie/baselines` | 0.37 GB |
  | `g3_pie/cache` | 1.5 GB(evidence) |
  | `envs/pie` | 15 GB(uv 캐시 7.4 GB 포함) |

---

## 8. 미완료·차단·주의

- **차단 없음.** HF 인증은 필요하지 않았다. 다운로드는 예산 안이었고, GPU도 할당받았다.
- **bit-exact 재현 아님.** 저자 하드웨어(H100)가 없다. 근접 일치 기준은 §2에 명시했다.
- **CPU 추론(job 20400)은 완료했다.** 3개 조건 모두 close였지만 AUPRC가 GPU·공개값보다 0.002–0.005 낮았다. 이는 장치 민감도 보조 자료이며 결론에 쓰지 않았다.
- **이번 범위 밖(미실행):** PIE 재학습, State/TxPert 평가, 사후 크기 보정 실험(G4 후보), HepG2 외 3 fold의 F3 Bs형 비교.
- **비교 한계:**
  - PIE의 이중 OOD 표적 집합(386)은 F3의 모의 이중 OOD(Bs, 무작위 20%)와 다르다. 그래서 F3 Bs 수치와 직접 비교하지 않았다.
  - bootstrap은 표적 단위이고, 표적은 경로·control을 공유하므로 독립이 아니다. 따라서 구간은 이 표적 집합에 대한 기술적 요약이다.
  - 기준선 ridge는 λ 격자 경계(10⁴)에서 선택되었다(소스 CV 기준). 더 큰 λ로 가면 평균 예측에 더 가까워질 뿐이라 결론은 바뀌지 않는다.

## 9. 보충: 다른 3 fold의 기준선 (같은 코드, bootstrap 500회)

보충 작업(20478–20483, 모두 완료)의 결과는 `reports/feedback/G3/pie_split_summary_<fold>.tsv`, `f3metrics_summary_<fold>.tsv`, 교차 요약 [pie_split_allfolds_summary.tsv](G3/pie_split_allfolds_summary.tsv)(`src/feedback/g3_pie_allfolds.py`)에 있다. PIE는 GPU 재현 예측이다. 기준선은 조건별로 공식 AUPRC가 가장 높은 것(같은 선택 규칙)을 골랐다. 이 보충은 HepG2 판정의 **일반성 확인용**이며, 선택에 쓰지 않았다.

| fold | 조건 | 기준선 | AUPRC PIE / 기준선 | AUPRC_x PIE / 기준선 | 방향_x PIE / 기준선 | PIE 표적 pct | pert-c r PIE / ridge | Δ RMSE PIE / no-change | PIE+규칙 AUPRC |
|---|---|---|---|---|---|---:|---|---|---:|
| hepg2 | ctx | same_target_cal | **0.442** / 0.391 | **0.331** / 0.282 | **0.752** / 0.741 | 0.98 | 0.311 / 0.118 | 0.102 / **0.086** | 0.452 |
| hepg2 | pert | context_mean_kd | 0.321 / **0.414** | **0.324** / 0.295 | **0.725** / 0.665 | 0.69 | 0.269 / 0.111 | 0.092 / **0.081** | 0.441 |
| hepg2 | ctx_pert | ridge_stacked_kd | 0.244 / **0.382** | 0.245 / 0.240 | **0.663** / 0.623 | 0.67 | 0.209 / 0.144 | 0.112 / **0.090** | 0.387 |
| jurkat | ctx | same_target_cal | **0.376** / 0.357 | **0.285** / 0.261 | **0.753** / 0.710 | 0.99 | 0.280 / 0.063 | 0.087 / **0.076** | 0.382 |
| jurkat | pert | ridge_inline_kd | 0.326 / **0.413** | **0.322** / 0.299 | **0.728** / 0.677 | 0.74 | 0.284 / 0.135 | 0.091 / **0.084** | 0.435 |
| jurkat | ctx_pert | global_mean_kd | 0.214 / **0.328** | **0.210** / 0.198 | **0.670** / 0.607 | 0.74 | 0.160 / 0.069 | 0.096 / **0.079** | 0.340 |
| k562 | ctx | same_target_cal | **0.422** / 0.386 | **0.306** / 0.272 | **0.739** / 0.666 | 0.97 | 0.295 / 0.089 | 0.099 / **0.064** | 0.443 |
| k562 | pert | context_mean_kd | 0.303 / **0.411** | **0.305** / 0.294 | **0.708** / 0.674 | 0.68 | 0.253 / 0.112 | 0.102 / **0.086** | 0.420 |
| k562 | ctx_pert | global_mean_kd | 0.198 / **0.358** | **0.200** / 0.178 | **0.631** / 0.545 | 0.72 | 0.160 / 0.085 | 0.105 / **0.062** | 0.378 |
| rpe1 | ctx | same_target_cal | **0.421** / 0.386 | **0.371** / 0.342 | 0.762 / **0.768** | 0.99 | 0.267 / 0.081 | 0.107 / **0.100** | 0.424 |
| rpe1 | pert | context_mean_kd | 0.269 / **0.399** | **0.271** / 0.249 | **0.704** / 0.636 | 0.72 | 0.255 / 0.142 | 0.091 / **0.077** | 0.421 |
| rpe1 | ctx_pert | ridge_stacked_kd | 0.290 / **0.379** | 0.292 / **0.316** | **0.696** / 0.678 | 0.73 | 0.141 / 0.112 | 0.117 / **0.100** | 0.352 |

- **HepG2에서 본 구조는 4 fold 모두 반복된다.**
  - 공식 AUPRC: 미관측 세포주는 PIE 우세(4/4). 미관측 perturbation·이중 OOD는 기준선 우세(4/4).
  - 미관측 perturbation에서 PIE의 표적 p_de 백분위는 0.67–0.74다(4/4).
  - Δ RMSE는 no-change보다 나쁘다(12/12).
  - 표적 제외 AUPRC는 11/12에서 PIE가 같거나 높다(RPE1 이중 OOD만 0.292 대 0.316).
  - pert-centred r(Δ)은 12/12에서 PIE > ridge다.
- **PIE + 표적 규칙(진단):** 이중 OOD에서 3/4 fold만 기준선을 넘는다(RPE1 0.352 대 0.379). 이중 OOD의 DE 집합 복원 이득은 fold에 따라 없거나 음수다.
- **F3 지표(우리 원자료 정답):**
  - 미관측 세포주에서 PIE Δ는 same_target_cal 대비 다음과 같다.
    - RMSE: 4/4 나쁨.
    - top-20 방향: 4/4 나쁨(0.63–0.73 대 0.80–0.84).
    - retrieval: 4/4 나쁨(0.100–0.179 대 0.077–0.128).
    - Pearson·DEG AUPRC: Jurkat·K562는 비슷하거나 약간 높고, HepG2·RPE1은 낮다.
  - 이중 OOD에서는 PIE가 4/4 fold에서 pert-centred r(0.15–0.23 대 0.07–0.16)과 retrieval(0.26–0.31 대 0.39–0.43)이 ridge보다 낫다. RMSE와 방향은 4/4 나쁘다.
  - 결론적으로 **HepG2가 특별히 나쁘거나 좋은 fold는 아니다.**

## 10. 재현 명령 (G1 worktree 루트에서)

```bash
# 0) 환경 (한 번)
/usr/local/anaconda3/bin/conda create -y -p assets/envs/pie/py312 -c conda-forge --override-channels python=3.12 uv
git clone https://github.com/ArcInstitute/pie.git assets/g3_pie/pie_code && git -C assets/g3_pie/pie_code checkout v1.0.1
(cd assets/g3_pie/pie_code && UV_PROJECT_ENVIRONMENT=$PWD/../../envs/pie/venv UV_CACHE_DIR=$PWD/../../envs/pie/uv_cache \
   ../../envs/pie/py312/bin/uv sync --frozen --python ../../envs/pie/py312/bin/python)
# 1) 자산 (약 48 GB) — 끝난 뒤에 평가를 제출할 것
sbatch experiments/feedback/g3_pie_fetch.sbatch
# 2) 공식 평가 (GPU 1장, 4 fold x 3 조건, 예측 저장) / CPU 대안
sbatch experiments/feedback/g3_pie_eval.sbatch cuda hepg2 jurkat k562 rpe1
sbatch --partition=cpu --gres=none --exclude=g[1-4] --cpus-per-task=32 --mem=128G experiments/feedback/g3_pie_eval.sbatch cpu hepg2
source experiments/feedback/g3_pie_env.sh && $PIE_PY src/feedback/g3_pie_repro_compare.py \
  --models assets/g3_pie/models/replogle_wdataset --runs assets/g3_pie/runs --out reports/feedback/G3
# 3) 기준선 (PIE split, pie.metrics) → F3 지표
sbatch experiments/feedback/g3_pie_baselines.sbatch hepg2 cuda
sbatch experiments/feedback/g3_pie_f3metrics.sbatch hepg2
```

## 11. 산출물

- **코드:**
  - `src/feedback/g3_pie_fetch.py`, `g3_pie_baselines.py`, `g3_pie_repro_compare.py`, `g3_pie_f3metrics.py`, `g3_pie_allfolds.py`
  - `experiments/feedback/g3_pie_env.sh`, `g3_pie_{fetch,eval,baselines,f3metrics}.sbatch`
- **보고서 파일 (`reports/feedback/G3/`):**
  - `provenance_fetch.json`, `env_freeze_pie.txt`, `slurm_jobs.tsv`
  - `repro_vs_published{,_4fold,_pairs}.tsv`
  - `pie_split_{summary,paired_bootstrap,strata,selection}_hepg2.*`, `pie_split_paired_bootstrap_kd_diag_hepg2.tsv`
  - `f3metrics_{summary,paired_bootstrap,meta}_hepg2.*`
  - (보충) `pie_split_*_{jurkat,k562,rpe1}.*`, `f3metrics_*_{jurkat,k562,rpe1}.*`, `pie_split_allfolds_summary.tsv`
- **G1 자산 (git 제외, `assets/g3_pie/`):**
  - `models/`(체크포인트·공개 CSV), `data/hf/`(고정 revision 자료·지식원·split)
  - `runs/{cuda,cpu}/replogle_wdataset/<fold>/eval/<setting>/`(재현 지표 + `predictions_best_auprc.parquet`)
  - `baselines/`(기준선 예측 npz, per-pair 지표, F3 per-target 지표, 정답 일치도), `cache/evidence/`

**상태:**

| 상태 | 항목 |
|---|---|
| 실측 완료 | G1-a 4 fold 재현(GPU), G1-b HepG2 기준선·bootstrap·층화, F3 지표 평가 |
| 진행 중 | 없음(§9 보충 결과 포함) |
| 미실행 | PIE 재학습, 다른 후보 모델, 크기 보정 실험 |
| 차단 | 없음 |
