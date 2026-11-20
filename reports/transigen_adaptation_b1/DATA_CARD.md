# 연구 B · 단계 B1 — 데이터 카드 (DATA_CARD)

**작성일:** 2026-11-16 · **protocol_variant:** `cmap2020_level3_multi_target_v1` (CMap 2020 Level 3, 다중 target 경로. GSE92742 fallback은 쓰지 않았다)
**규칙 파일:** `configs/transigen_adaptation_b1/protocol.yaml` — 대상 세포주의 발현 값을 읽기 전에 commit했다(`a26adaf`).
**원자료:** `manifests/transigen_adaptation_b1/`의 `download_manifest.json`, `attrition.tsv`, `build_manifest.json`, `splits_summary.json`, `splits/*.tsv`, `source_split.tsv`, `source_cells.tsv`, `target_rows.tsv`, `genes_landmark_978.tsv`, `dropped_no_structure.tsv`, `source_rows_excluded_degenerate.tsv`

## 1. 무엇을 받았는가

서버에는 CMap 2020의 Level 5만 있었다. 공식 배포처(clue.io의 CMap2020 자료 페이지가 가리키는 공개 S3 저장소)에서 로그인이나 결제 없이 Level 3를 받을 수 있었다. Slurm CPU 작업 하나(job 23927)로 한 번만 받았다.

| 파일 | 크기 (bytes) | SHA-256 | 무결성 |
|---|---:|---|---|
| `level3_beta_trt_cp_n1805898x12328.gctx` | 89,057,706,488 | `d8ce15064cfccd817e322732717ee3ad0e3294c55cd5276286860810721dfe28` | S3 multipart ETag 일치 |
| `level3_beta_ctl_n188708x12328.gctx` | 18,611,750,206 | `45f528c978d48d8ac320f184a8d0cb21f22d54582d7af24b50701a91b6370d85` | S3 multipart ETag 일치 |
| `instinfo_beta.txt` | 674,838,120 | `2676189e8ea4a4cfdafc6c7e1b9f1776f4459ebfe4f79f6a736c3f3a732d3249` | S3 multipart ETag 일치 |

- 받은 날짜는 2026-11-16이다. URL은 `download_manifest.json`에 있다.
- 받는 동안 임시 파일(`.part`)에 쓰고, 크기가 맞을 때만 이름을 바꿨다. 동시에 두 번 받지 못하도록 잠금 파일을 썼다.
- 유전자·세포주·화합물 주석은 서버에 이미 있던 같은 release의 `geneinfo_beta.txt`, `cellinfo_beta.txt`, `compoundinfo_beta.txt`를 읽기 전용으로 썼다.
- Level 5 행렬과 저자 저장소의 ID 전용 파일은 발현 입력으로 쓰지 않았다. GSE92742·GSE70138과 합치지 않았다.

## 2. 한 행이 무엇인가

```text
한 행 = (세포주, 화합물 pert_id, detection plate)
X0 = 그 plate에 있는 같은 세포주의 DMSO well 평균 (산술평균)
X1 = 그 plate에 있는 그 화합물 well 평균 (균등평균)
조건 = (세포주, 화합물).  정답 Δ = plate마다 구한 (X1 − X0)의 균등평균
```

- 값은 공식 Level 3 그대로다. 추가 정규화나 scaler를 신경망 입력에 쓰지 않았다. 만들어진 X0·X1의 전체 평균은 8.44와 8.43, 범위는 0–15.0이다.
- 유전자는 `geneinfo_beta.txt`에서 `feature_space == landmark`인 978개이며, 파일에 나온 순서로 고정했다. 추론된 유전자는 쓰지 않았다.
- 약물 표현은 `compoundinfo_beta.txt`의 SMILES로 만든 ECFP4(Morgan 반경 2) 2,048 bit다.
- **약물 그룹**: 이름(cmap_name), InChIKey, RDKit canonical SMILES, 지문 가운데 하나라도 같은 pert_id를 한 그룹으로 묶었다. 분할은 그룹 단위로 한다.

## 3. 선택 규칙과 남은 수

| 단계 | well 수 | 세포주 | pert_id | HUH7 well / pert_id | JHH5 | JHH7 | HUH751 |
|---|---:|---:|---:|---|---|---|---|
| 단일 저분자(trt_cp) 전체 | 1,805,898 | 230 | 34,419 | 3,502 / 771 | 2,152 / 725 | 363 / 120 | 85 / 28 |
| 24 h | 1,338,831 | 168 | 23,347 | 759 / 135 | 2,152 / 725 | 363 / 120 | 85 / 28 |
| `nearest_dose` 10 µM | 352,376 | 167 | 11,962 | 318 / 110 | 708 / 704 | 113 / 113 | 33 / 11 |
| `qc_pass == 1` | 260,691 | 167 | 11,824 | 179 / 90 | 663 / 660 | 111 / 111 | 30 / 11 |
| 같은 plate·같은 세포주의 qc 통과 DMSO well 있음 | 258,534 | 165 | 11,820 | 179 / 90 | 663 / 660 | 111 / 111 | 30 / 11 |
| 구조(SMILES)를 읽을 수 있음 | 247,301 | 164 | 10,861 | 179 / 90 | 663 / 660 | 111 / 111 | 0 / 0 |
| HUH751 제외 | 247,301 | 164 | 10,861 | 179 / 90 | 663 / 660 | 111 / 111 | — |

- 농도는 release가 제공하는 명목 농도 구간 `nearest_dose`가 10 µM인 well로 정했다. 실제 분주 농도 `pert_dose`에는 10.0001, 9.9785, 10.4 같은 오차가 있어서다. 이 규칙은 발현 값을 보기 전에 정했다. 이전 감사(Level 5 signature 기준, `pert_dose`가 정확히 10)의 약물 수 73 / 679 / 103과 다른 것은 기준 열과 자료 수준이 다르기 때문이다.
- `qc_pass`는 release가 well마다 붙인 기술적 품질 표시다. 처리군과 대조군에 똑같이 적용했다. 약물 반응의 크기나 반복 간 상관으로 고른 것이 아니다.
- HUH7은 10 µM·24 h well 318개 가운데 139개가 `qc_pass == 0`이라 빠졌다.
- 구조가 없는 화합물 959개는 뺐다(`dropped_no_structure.tsv`). HUH751의 화합물 11개는 모두 구조가 없었다. HUH751은 Huh-7의 파생 clone이므로 구조가 있었더라도 source와 target 어디에도 넣지 않는 규칙이다.

만들어진 행: **source 238,574행**(161개 세포주, 87,975개 조건), **target 953행**. 약물은 10,861개 pert_id, 8,288개 약물 그룹이다.

## 4. Target (성인 HCC 세 계열)

| target | plate | 조건 (pert_id) | 약물 그룹 | 조건당 plate 수 | plate당 DMSO well | 판정 |
|---|---|---:|---:|---|---|---|
| HUH7 (Huh-7) | `CVD001_HUH7_24H_X1/X2/X3` 3장 | 90 | 69 | 1장 27개, 2장 37개, 3장 26개 | 12, 13, 14 | 적격 (40 그룹 이상) |
| JHH5 (JHH-5) | `MOAR008–012`, `ASG002` 6장 | 660 | 652 | 1장 657개, 2장 3개 | 9–14 | 적격 |
| JHH7 (JHH-7) | `ASG002_JHH7_24H_X1` 1장 | 111 | 111 | 모두 1장 | 13 | 적격 |

- 모든 처리 행은 well 1개의 측정이다(plate 안 반복 없음).
- 세 target 모두 측정된 약물 그룹 전부가 source 학습 세포주에도 10 µM·24 h 조건으로 있다(빠진 그룹 0개). 따라서 주 평가는 "source에서 관측한 약물을 그 HCC 세포주에서 예측"하는 과제다.
- 약물 그룹이 source 학습 세포주 몇 개에서 관측됐는지(1번 분할 기준): HUH7 최소 1 · 중앙값 8 · 최대 115, JHH5 최소 4 · 중앙값 28 · 최대 115, JHH7 최소 14 · 중앙값 81 · 최대 115.
- HUH7은 14개 약물 그룹이 pert_id를 2개 이상(최대 5개) 가진다. JHH5는 8개 그룹이 2개씩이다. 같은 그룹은 항상 같은 분할에 들어간다.
- JHH7은 plate가 한 장이라 모든 조건의 X0가 같다. 이 세포주에서 모델이 조건을 구별할 수 있는 입력은 약물 지문뿐이다.
- JHH5와 JHH7의 대부분 조건은 독립 반복이 없다. 정답 Δ 자체의 측정 잡음을 이 세포주들에서는 추정할 수 없다.

### 분할 (약물 그룹 단위 60/20/20, 분할 seed 1·2·3)

| target | 적응 후보(adapt pool) | 검증(val) | 시험(test) | 주 k | 학습곡선 k (1번 분할만) |
|---|---:|---:|---:|---:|---|
| HUH7 | 41 | 14 | 14 | 16 | 8, 32 |
| JHH5 | 392 | 130 | 130 | 16 | 8, 32 |
| JHH7 | 67 | 22 | 22 | 16 | 8, 32 |

- 숫자는 약물 그룹 수이며 세 분할에서 같다. k개 적응 약물은 pool에서 미리 정한 순서의 앞 k개다(k=8 ⊂ 16 ⊂ 32).
- **실제로 필요한 HCC 실측 라벨은 k개 적응 약물과 검증 약물을 합친 것이다.** 주 설정에서 HUH7은 16 + 14, JHH5는 16 + 130, JHH7은 16 + 22 약물 그룹이다. 검증 라벨은 학습률과 중단 시점 선택에 쓰인다.
- B1과 B2는 같은 적응 행·같은 검증 행·같은 시험 행을 쓴다. 실행마다 행 목록의 hash를 기록하고 두 arm이 같은지 확인한다.

## 5. Source

- 161개 세포주, 238,574행. 대상 HCC 계열(HUH7, HUH751, JHH5, JHH7)은 모든 조건에서 뺐다.
- 간 계열이지만 성인 HCC가 아닌 HEPG2(간모세포종, 4,150행·1,357조건)와 PHH(일차 간세포, 297행·239조건)는 source에 들어 있다. 성인 HCC 자료로 세지 않는다.
- 행이 많은 세포주: PC3 36,435, MCF7 30,777, VCAP 19,520, A549 16,358, A375 13,793, HA1E 13,386, HT29 10,969.
- **source 검증**: 세포주의 10%인 16개를 seed로 뽑아 통째로 떼어 놓았다(22RV1, BC3C, CW2, HCC515, HFL1, HIMG002, MCF10A, MINO, NCIH596, OCILY3, SKMEL5, TC32, XC.P091, XC.P092, XC.P910, XC.P911). 학습 145개 세포주 221,100행, 검증 17,432행이다.
- source의 학습률과 epoch는 이 source 검증으로만 고른다. HCC 자료는 쓰지 않는다.

## 6. 규칙 변경 1건 (amendment 1)

- **관측:** 처음 실행한 source 학습 3개(`source_lr*_seed42_v1`)가 모두 첫 epoch 안에서 loss가 `nan`이 되어 멈췄다. update마다 추적하니 한 batch에서 loss가 무한대가 됐다. source 행의 발현 평균을 보니 plate `REP.A010_JURKAT_24H_X3_B32`의 42행은 X0와 X1이 모두 약 0.001이었다. 다른 행의 평균은 7.1–9.5이고 그 사이에는 아무 행도 없다. 이 plate는 `qc_pass == 1`이며 내려받은 파일의 ETag는 공식 파일과 일치한다.
- **조치:** X0 또는 X1의 978개 평균이 1 미만인 **source 행**을 source 학습·검증·기준선에서 뺐다(42행, `source_rows_excluded_degenerate.tsv`). target 행은 하나도 거르지 않았다.
- **그때 본 것:** source 행 평균, 발산한 source 실행 3개, target의 X0 행 평균(7.95–9.02). target의 X1 값, 적응 실행, 검증·시험 점수는 보지 않았다.
- **확인:** 42행을 뺀 뒤 학습률 1e-3으로 한 epoch(약 3,400 update)를 추적했을 때 loss가 유한했다. 학습률 3e-3은 42행을 빼도 28번째 update에서 발산했다. 이 후보는 실패한 후보로 그대로 기록한다.
- 발산한 v1 실행은 지우지 않고 run ledger에 남겼다. 다시 실행한 것은 `_v2` ID를 쓴다. 학습률 후보와 예산은 바꾸지 않았다.
- **가설(검증 안 함):** 이 plate의 값이 왜 0.001인지는 확인하지 않았다. release 가공 과정의 결측 표시일 수 있다.

## 7. 확인하지 못한 것

- 저자 논문의 MODZ 집계와 수치가 같은지는 비교하지 않았다. 이 연구의 집계는 계획서의 단순 표준 집계다.
- `qc_pass == 0` well을 포함했을 때의 결과는 실행하지 않았다.
- HUH7의 CVD001 plate는 GSE92742에도 같은 측정이 들어 있다. 이 연구는 CMap 2020 한 release만 쓰므로 중복으로 세지 않았지만, 두 release의 값이 같은지는 비교하지 않았다.
