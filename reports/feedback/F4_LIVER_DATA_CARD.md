# F4 — 간 관련 perturbation 자료 데이터 카드 (화학적 HCC 경로 중심)

**작성:** 2026-11-10 · 계획 [FEEDBACK_PLAN_20261110.md](../../docs/FEEDBACK_PLAN_20261110.md) §4 · 소유 경로 `reports/feedback/F4/`
**성격:** 실측(이번 작업에서 Slurm으로 계산한 값), 웹 확인(원 출처 메타데이터 확인), 추정(계산 근거를 밝힌 예상치), 미실행을 구분해 표기한다. 모델 학습이나 예측은 하지 않았다.

## 0. 결론 요약

| 항목 | 결과 |
|---|---|
| SNU-423 (Tahoe-100M) | 약물 **379개 × 용량 3개(0.05/0.5/5 µM) = 1,137개 drug×dose**, plate 14개, 처리 well 1,316개, DMSO well 28개(plate당 2개). 세포 1,501,700개(Tahoe `pass_filter=full`), DMSO 세포 36,803개(plate당 1,152–5,016개). **용량은 `sample_metadata.drugname_drugconc`로 복원된다.** |
| 반복 | drug×dose 975개는 well 1개뿐이다. 161개는 well 2–3개인데 대부분 서로 다른 plate에 있다. Adagrasib 0.05 µM은 16개 well(14개 plate 전부)에 들어 있다. **약물 379개 중 354개는 세 용량이 서로 다른 plate에 놓여, 용량과 plate가 교란된다.** |
| 효과 강도 (실측) | DMSO pseudo-well로 맞춘 sampling 잡음 모델과 실제 DMSO well 쌍(well 간 효과 포함)을 null로 썼다. 처리 well의 **31.5%(414/1,316)**가 모든 DMSO well 쌍보다 큰 변화를 보인다. 이를 만족하는 drug×dose는 393개, 약물은 255개(67%)다. 그 두 배 기준(strong)을 넘는 well은 7.0%(drug×dose 90개)다. 처리 well의 중앙값 변화는 DMSO well끼리의 전형적 차이의 1.66배에 그친다. |
| 재현성 | 대조 약물을 뺀 반복 well 쌍 167개의 log2FC 상관 중앙값은 **0.41**이다. 서로 다른 약물·plate 쌍의 null은 0.13(95분위 0.40)이다. 0.5 µM과 5 µM 사이의 용량 일치도는 0.42다. |
| 판정 | **SNU-423: 조건부 적격**(pseudobulk drug×dose 단위의 leave-one-cell-line-out 평가). 단, 검출 가능한 조건으로 층화하고, 같은 well의 다른 세포주 평균을 기준선으로 의무화하고, 용량–반응 해석은 금지한다. **HepG2/C3A (Tahoe): 조건부 적격(간모세포종 proxy).** |
| 전제 정정 | **나머지 48개 Tahoe 세포주는 이미 NFS에 있다.** `/mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M`(HF 미러)에 발현 샤드 3,388/3,388개(337.6 GB), obs_metadata, 공식 pseudobulk DE(1,026 샤드, 88.9 GB)가 있다. **R3에 필요한 신규 다운로드는 0 GB**이고, 파생 추출물만 2.7–24 GB 생긴다(§6). |
| 신규 간 후보 | 성인 HCC 세포의 단일세포 perturbation 자료는 Tahoe(100M, EmeraldBay) 말고는 찾지 못했다. 그 밖에는 HepaRG CRISPRi(GSE238219, 표적 11개), HepG2 CROP-seq(GSE281160), HepaRG HTTr bulk(GSE284321, 화학물질 1,201개; GSE296254) 정도이며 모두 조건부다(§5). |

## 1. 전제 정정: Tahoe-100M 전체가 로컬에 있다

- 작업 지시에는 "SNU-423과 HepG2-C3A만 디스크에 있다"고 되어 있었다. 실제 `/mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M`(= g1 `/data/pubdata/Tahoe/Tahoe-100M`, 소유자 `hg`, 그룹 sysmed 읽기 가능)에는 HF 저장소 전체가 있다.
  - 발현 샤드 3,388개, 337,644,770,670 B. HF 카드의 `download_size`와 바이트 단위로 일치한다(디렉터리 엔트리 139,264 B 제외).
  - `metadata/`: obs_metadata(100,648,790행), sample/drug/cell_line/gene 메타데이터, `pseudobulk_differential_expression/` 1,026 샤드(88.9 GB).
- 샤드 하나에는 plate 하나(50개 세포주 전부, well 약 60개, 세포 28,225개)가 들어 있다. 따라서 특정 세포주만 받는 부분 다운로드는 불가능하고, plate 단위 부분 다운로드만 가능하다(plate당 약 24 GB).
- 이 미러는 다른 사용자의 원본 계층이다. 읽기만 했고 아무것도 쓰지 않았다. 다른 사용자 데이터를 정식 학습 입력으로 쓰기 전에는 소유자 확인을 권장한다.
- 디스크의 `by_cell_line/SNU-423_counts.h5ad`는 obs_metadata의 SNU-423 `pass_filter=full` 세포 1,501,700개와 정확히 일치한다. `minimal` 세포 44,364개는 제외돼 있다. HepG2/C3A도 `full` 1,244,192개로 같은 방식이다.

## 2. 데이터 카드 — Tahoe-100M SNU-423 (CVCL_0366, 성인 HCC 유래)

| 항목 | 내용 (실측: job 20325/20327/20339) |
|---|---|
| **접근성** | CC0. 로컬 원시 count h5ad 6.1 GB(1,501,700 × 62,713, CSR gzip). `var`에 ensembl_id는 없지만 index 위치로 token_id를 복원할 수 있다. 전체 sha256은 ACQUISITION 문서에 있다. |
| **대조군** | `DMSO_TF`, plate당 2개 well, 합계 36,803개 세포. plate별 1,152(plate13)–5,016(plate2)개. 처리 well과 같은 plate의 DMSO를 쓸 수 있다(plate-matched). |
| **개입** | 약물 379개(PubChem CID, SMILES, MoA 27종, 표적 주석), 용량 0.05/0.5/5 µM으로 drug×dose 1,137개. 시점은 하나다(프리프린트 기준 24 h, 데이터에서는 확인하지 못함). 96-well plate 14개, plate13에는 약물 43개만 있다. |
| **측정** | Parse 단일세포 RNA(MOSAIC: 50개 세포주를 한 well에 함께 배양). 처리 well당 세포는 중앙값 1,065개(IQR 713–1,434, 최소 1, 최대 3,218)다. 50개 미만 well이 15개, DMSO의 20% 미만인 well이 29개다. 세포당 평균 검출 유전자는 1,796개, CPM≥10 발현 유전자는 9,950개다. QC 필드: gene/tscp/mread count, pcnt_mito, S/G2M score, phase(G1 1.14M, G2M 0.36M, S 0.05M), pass_filter. |
| **반복** | drug×dose 기준으로 975개는 well 1개, 158개는 2개(152개는 서로 다른 plate), 3개는 3개, Adagrasib 0.05 µM은 16개다. 약물 기준으로는 246개가 3개 plate, 112개가 4개 plate에 있다. **세 용량이 한 plate를 공유하는 약물은 25개뿐이다**(354개는 용량이 서로 다른 plate에 있다). 이 반복은 생물학적 독립 반복이 아니라 같은 실험 안의 다른 well이다. |
| **plate 교란 구조** | (i) 용량이 plate와 교란된다. (ii) 50개 세포주가 같은 well을 공유하므로, 학습 세포주와 SNU-423이 동일한 well·plate 기술 효과를 함께 갖는다. (iii) DMSO pool끼리 plate 간 차이(rms log2FC 중앙값 0.357)가 같은 plate 안 DMSO well 간 차이(0.276)보다 크다. 따라서 plate-matched 대조가 필수다. |
| **분할 가능성** | SNU-423의 처리 well 전체를 보류하는 leave-one-cell-line-out은 가능하다. 다른 48개(또는 49개) 세포주가 같은 drug×dose를 모두 갖는다. SNU-423 내부의 미관측 약물 분할도 가능하다. 용량 외삽은 plate 교란 때문에 해석 불가로 둔다. |
| 산출물 | [`F4/snu423_drug_dose_plate_summary.tsv`](F4/snu423_drug_dose_plate_summary.tsv)(well 1,344행: 세포 수, DMSO 대비 비율, QC, 표적, MoA), [`F4/snu423_effect_strength.tsv`](F4/snu423_effect_strength.tsv), [`F4/snu423_replicate_pairs.tsv`](F4/snu423_replicate_pairs.tsv), [`F4/tahoe_design_summary.json`](F4/tahoe_design_summary.json) |

### 2-1. 효과 강도 분석 방법 (실측)

1. **pseudobulk:** SNU-423 h5ad를 한 번 스트리밍하면서 well(sample)별 원시 count 합을 만들었다. 같은 pass에서 plate마다 DMSO 세포를 무작위로 B=2/4/8/16/32개 pseudo-well로 나눈 합도 만들었다(job 20327, 4 CPU, 241 s).
2. **변화량:** 같은 plate DMSO 두 well의 합을 기준으로 삼았다. 이 기준 대비 log2(CPM+1) 차이를 DMSO CPM≥10인 9,950개 유전자에서 계산했다.
3. **잡음 모델:** 평균 제곱 변화량을 h=1/UMI₁+1/UMI₂의 거듭제곱 함수로 맞췄다(DMSO pseudo-well vs 나머지 DMSO, 즉 순수 sampling 잡음). 적합 후 pseudo-well의 비율 R의 중앙값은 B와 무관하게 0.93–1.18이었다. 처리 well은 R = 관측 rms² / sampling 기대값으로 요약했다.
4. **null:** 같은 plate의 실제 DMSO well 쌍 14개는 sampling에 well 효과까지 포함한다. 이들의 R은 중앙값 2.03, 최대 4.31이었다. **"검출 가능"은 R > 4.31(모든 DMSO well 쌍 초과), "strong"은 R > 8.62로 정의했다.** 처리 well 대 DMSO 두 well 합의 비교는 well 효과를 약 1.5배 포함하고 DMSO 쌍은 2배 포함하므로, 이 기준은 보수적이다.
5. 처음 시도한 두 null 정의(크기 맞춤 bucket 분위수, 선형 잡음 모델)는 job 20335/20336에서 각각 plate 혼합과 모델 오지정이 확인되어 폐기했다. 최종값은 job 20339의 결과다.

### 2-2. 결과

| 지표 (처리 well 1,316개) | SNU-423 | HepG2/C3A (같은 well) |
|---|---:|---:|
| R(sampling 대비) 중앙값 [IQR] | 3.37 [2.36–4.79] | 2.06 [1.50–2.99] |
| 전형적 DMSO well 쌍 대비 배수 중앙값 | 1.66 | 1.38 |
| DMSO well 쌍 R 중앙값 / 최대 | 2.03 / 4.31 | 1.50 / 2.88 |
| 검출 가능 well 비율 | **31.5%** | 27.4% |
| 검출 가능 drug×dose(≥1 well) / 약물(임의 용량) | **393 / 255 (67.3%)** | 323 / 209 (55.1%) |
| 용량별 검출률 0.05 / 0.5 / 5 µM | 21.2% / 39.3% / 33.9% | 19.9% / 29.1% / 32.5% |
| strong well 비율 / drug×dose / 약물 | 7.0% / 90 / 70 (18.5%) | 6.5% / 85 / 74 (19.5%) |
| 세 용량에서 효과가 단조 증가하는 약물 | 30.3% | 25.1% |
| 세포 수 비율(세포독성 대리)과 효과의 Spearman | 0.06 | 0.08 |

- SNU-423에서 가장 강한 반응은 TAK-901, Ouabain, Homoharringtonine/Harringtonine, Panobinostat/Belinostat(HDAC), Dinaciclib였다. MoA로 보면 proteasome, HDAC, glucocorticoid receptor 작용제, MAPK/MEK, 단백질 합성 억제제 순이다.
- HCC 관련 약물: Regorafenib은 0.5 µM 두 well 중 한 well만 검출 가능했다(R 9.95 vs 1.69). 반복 well 사이에 불일치가 있다는 뜻이다. Cabozantinib은 5 µM에서만 검출됐다. Sorafenib과 Lenvatinib은 Tahoe 약물 목록에 없다.
- **세포 수 감소와 전사 효과는 거의 무관하다(ρ≈0.06).** 세포 비율은 세포독성 또는 증식의 대리 지표일 뿐 반응 강도를 대신하지 못한다.

### 2-3. 반복·용량·세포주 간 일치도 (실측)

| 비교 | SNU-423 | HepG2/C3A |
|---|---:|---:|
| 반복 well 쌍(Adagrasib 대조 제외) 수 / log2FC r 중앙값 | 167 (drug×dose 161) / **0.41** | 165 / 0.37 |
| 그중 두 well 모두 검출 가능한 쌍: 수 / r 중앙값 | 19 / 0.50 | 36 / 0.49 |
| Adagrasib 0.05 µM 대조 120쌍의 r 중앙값 (검출 well) | 0.27 (3/18) | 0.16 (2/18) |
| null: 다른 약물·다른 plate 쌍의 r 중앙값 (95분위) | 0.13 (0.40) | 0.12 (0.32) |
| 0.5 vs 5 µM 같은 약물의 r 중앙값 (서로 다른 plate 365/379) | 0.42 | 0.33 |
| 같은 well의 SNU-423 vs HepG2/C3A r 중앙값 / 같은 plate·다른 약물 null | 0.42 / 0.19 (95분위 0.39) | — |
| 둘 다 검출 가능한 well 286개의 r 중앙값 | 0.56 | — |
| 검출 일치표 (둘 다 / SNU만 / HepG2만 / 둘 다 아님) | 286 / 128 / 74 / 826 | — |

같은 well의 SNU–HepG2 일치도에는 공통 well 기술 효과가 섞여 있어, 생물학적 공유 반응과 구분할 수 없다. LOCO에서 "다른 세포주 평균"과 "같은 well 다른 세포주" 기준선이 높게 나올 것을 예상해야 한다.

### 2-4. Tahoe 공식 pseudobulk DE와의 대조 (job 20326)

- 50개 세포주의 조건 65,218개를 요약했다([`F4/tahoe_pbde_condition_summary.tsv.gz`](F4/tahoe_pbde_condition_summary.tsv.gz), [`F4/tahoe_pbde_line_summary.tsv`](F4/tahoe_pbde_line_summary.tsv)).
- SNU-423의 조건당 padj<0.05 유전자 수 중앙값은 1,082개(50개 세포주 중 23위), HepG2/C3A는 451개(38위)다. 모든 SNU-423 조건이 10개 이상, 97%가 100개 이상이다. 거의 효과가 없는 약물(Entecavir 0.05 µM)도 52개다. 공개 DE에는 DMSO 대 DMSO 대조가 없다. **따라서 공식 padj는 null 보정이 안 된 값이며, 반응 유무를 거르는 필터로 쓰면 안 된다.** 순위로는 쓸 수 있다(우리 excess variance와 Spearman 0.84).
- 세포주 안 상대 기준(n_sig05_lfc05 ≥ 세포주 중앙값의 3배)을 쓰면 SNU-423 조건 128개(9.8%)가 해당하고, 그중 102개는 우리 기준으로도 검출 가능하다.
- **학습 세포주만으로 고른 조건이 SNU-423 반응을 예측하는가:** 비간 48개 세포주 중 10개 이상에서 strong인 drug×dose 232개(약물 160개)를 골랐다. 이 조건들의 SNU-423 well은 **61%가 검출 가능**했다(전체 기준 31.5%). 약물 활성의 세포주 간 전이 신호가 있다는 뜻이다(사후 기술 통계일 뿐, 이 선택에 SNU-423 정답은 쓰지 않았다).

## 3. 데이터 카드 — Tahoe-100M HepG2/C3A (CVCL_1098)

- **생물학:** HepG2의 C3A 하위클론이다. 원 HepG2는 15세 소아 간종양 유래로 현재 **간모세포종(hepatoblastoma)**으로 분류되며, 성인 HCC가 아니다. Tahoe-100M에서 간 기관 주석이 붙은 세포주는 SNU-423과 HepG2/C3A 둘뿐이다. SK-HEP-1은 cell_line_metadata에는 있지만 Tahoe-100M 세포는 0개다.
- **접근성·대조군·개입:** SNU-423과 같은 plate와 well에 있고 설계도 같다. 세포 1,244,192개(full), DMSO 29,046개(plate별 648–4,655개), 처리 well당 세포 중앙값 811개. 전처리본 `raw_data/data/HepG2_C3A_filtered_preprocessed.h5ad`(log1p, HVG 2,000개)와 세포가 동일하다.
- **측정 품질:** 세포당 유전자 수와 UMI가 SNU-423보다 적다. plate3(648개)과 plate7(687개)의 DMSO가 적어 그 두 plate의 잡음이 크다.
- **역할:** 두 번째 간 유래 보류 세포주 또는 검증용 세포주로 쓸 수 있다. 성인 HCC 근거로 제시해서는 안 된다. 판정: **조건부 적격(proxy)**.

## 4. 참고 카드 — GSE264667 HepG2 CRISPRi, GSE149614, EmeraldBay

| 자료 | 접근성 | 대조군 | 개입 | 측정 | 분할 가능성 | 판정 |
|---|---|---|---|---|---|---|
| **GSE264667** Nadig HepG2/Jurkat CRISPRi (로컬 5.6 GB HepG2 h5ad. scPerturb, HKU 균일 DE, vcc-uba G29도 같은 원 실험) | 공개 | non-targeting guide | 필수유전자 약 2,394개 | 단일세포 RNA | 미관측 표적, 세포주 간 전이 | **적격(유전적)**. 분석은 F3가 맡는다. 간모세포종이고, 기존 연구에서 노출됐으므로 새 blind cohort가 아니다. |
| **GSE149614** HCC 환자 10명 | 로컬 165 MB count + metadata | 없음 | 없음(관측 자료) | scRNA 71,915개 세포(Tumor/Normal/PVTT/LN, 세포유형 라벨) | perturbation 평가 불가 | **부적격**(perturbation 정답 없음). 세포 상태 참고용으로만 쓴다. |
| **Tahoe EmeraldBay** (로컬 57.7 GB, CC-BY-4.0; job 20329) | 공개 | `DMSO_TF` 4 well과 `DMSO_T0` | 단일 약물 20개(조건 46개)와 병용 47개 조건. 5일 배양. 성장률 판독 포함 | scRNA. SNU-423 40,423개 세포(DMSO 3,118개), 조건당 약 400개 세포. HepG2/C3A 116,500개 | 조건당 sample 1개(반복 없음). Tahoe-100M과 공유하는 약물 9개(Regorafenib, Paclitaxel, Gemcitabine, Irinotecan, Oxaliplatin, Lapatinib, Encorafenib, Adagrasib, RMC-6236), 용량과 시간은 다름 | **조건부**. 다른 실험에서 SNU-423 반응을 외부 확인하는 용도다. 동일 정답으로 쓰면 안 된다. |

## 5. 신규 및 로컬 간 관련 후보 인벤토리

전체 표는 [`F4/liver_perturbation_inventory.tsv`](F4/liver_perturbation_inventory.tsv)(18행)다. 검토 순서는 accession → 실제 발현 파일 → control 연결 → target/drug/dose/time/replicate → 개입 수 → 분할 → 저장·GPU → 판정이다. 웹 확인은 하위 에이전트가 GEO text/FTP, HF API, ENCODE API로 했다. GEO HTML은 captcha 때문에 text 형식으로 우회했다.

| 후보 | 세포계 / 증거 수준 | 개입·측정 | 대조·라벨 | 판정 |
|---|---|---|---|---|
| GSE238219 HepaRG NAFLD CRISPRi (Hepatology 2024; vcc-uba G23, 미다운로드 1 GB 미만) | HepaRG(HCC 유래 전구세포주, 간세포 유사 분화) | CRISPRi 11개 유전자, scRNA 17,076개 세포 | NTC, 세포별 guide call 있음 | **조건부** (표적 20개 미만, 미관측 표적 평가만 가능) |
| GSE281160 HepG2 MASLD 변이 CROP-seq (2026) | 간모세포종 | 비암호화 요소 20개 CRISPRi, scRNA | `cellAssign` 있음, NTC 미확인 | **조건부** (cis 효과가 약함) |
| GSE291757 HepG2 snHH-seq 약물 (2025) | 간모세포종 | 약물 11개와 병용 6개, 48 h, snRNA | 배지 대조(DMSO 없음), 조건 매핑 파일 없음 | **조건부** (라벨 복원 필요, 약물 수 적음) |
| GSE284321 EPA HTTr HepaRG (2025, 약 0.7 GB) | HepaRG | 화학물질 1,201개, 다중 농도, 24 h, **bulk** TempO-seq | plate별 DMSO | **적격(bulk 한정)**. scRNA 지표와 혼용 금지 |
| GSE296254 HTTr HepaRG Tox21 (로컬 135 MB) / GSE296248 PFAS | HepaRG | bulk, sample 약 8,600개 열, 농도 계열, block/plategroup/culture 배치 | vehicle·reference 화학물질 | **조건부(bulk)** |
| GSE308567 DILImap (Nat Commun 2025) | 1차 간세포(공여자 1명) | 300개 × 4용량 bulk. 공개 count는 51개 화합물 | DMSO | **조건부**(접근 제한) |
| LINCS L1000 HEPG2/HUH7, ENCODE HepG2 KD/KO | Huh7은 성인 HCC, HepG2 | bulk(L1000은 978 landmark 유전자) | vehicle/control | **조건부(bulk 참고)**. 세부는 일부 사전지식이며 미확인 |
| GSE206325, GSE151530/GSE125449 (환자, 면역항암 전후), Zenodo 17735506 (정상 간 공간전사체) | 환자 조직 | 통제된 개입 없음 | 짝지은 비처리 대조 없음 | **부적격** |
| OSCAR organoid sc-CRISPR (2023) | 마우스 담관 organoid | KO 약 79개 | NTC | **부적격(사람 아님)** |
| 비간 로컬 Perturb-seq(Replogle, X-Atlas Orion, KOLF2.1J, Feng iPSC, GSE295214, VCC H1, CD4T, GSE305037, vcc-uba G*) | 비간 | CRISPRi/KO | NTC | 간 평가로는 **부적격**. 유전적 전이의 학습원으로는 쓸 수 있다(F3) |

**핵심:** Huh7, Hep3B, PLC/PRF/5, SNU-449/475/387, HCCLM3, MHCC97H에서 Perturb-seq, CROP-seq, 다중화 약물 scRNA 공개 자료는 찾지 못했다. 웹 검색과 2023-06–2026 GEO 제목 105건을 선별했다. 이 범위 안의 결과이며 "어디에도 없다"는 주장은 아니다. **성인 HCC 세포의 단일세포 perturbation 정답은 현재 Tahoe(100M, EmeraldBay)의 SNU-423이 사실상 유일하다.** MIX-Seq(McFarland 2020)의 간 세포주 포함 여부는 확인하지 못했다.

## 6. 증거 수준 구분

| 수준 | HepG2 / HepG2-C3A | SNU-423 | 환자 HCC (GSE149614 등) |
|---|---|---|---|
| 기원 | 소아 간종양 → **간모세포종** 재분류 | **성인 HCC** 유래 세포주 (Cellosaurus CVCL_0366) | 성인 HCC 환자 조직 |
| 이용 가능한 개입 정답 | 유전적(CRISPRi, GSE264667), 화학적(Tahoe, EmeraldBay), bulk(ENCODE, L1000) | 화학적만 있음(Tahoe-100M 1,137 drug×dose, EmeraldBay 조건 91개) | 없음(관측 자료, 일부 임상 치료 전후) |
| 맥락 | 2D 단일 세포주 | MOSAIC 50개 세포주 공동배양(같은 well) | 종양 미세환경, 환자 이질성 |
| 주장 가능한 범위 | "간 유래 세포주로의 전이"까지 | "성인 HCC 유래 **세포주 1개**에서의 약물 전사 반응 예측"까지 | perturbation 예측 검증 불가. 상태 참조로만 사용 |
| 주장 불가 | 성인 HCC, 환자 약효 | 환자 약효·예후, HCC 일반화(세포주 1개, 단일 시점) | — |

## 7. R3 화학적 LOCO: 최소 확보 계획과 설계 조건

계획 세부는 [`F4/r3_minimal_acquisition_plan.json`](F4/r3_minimal_acquisition_plan.json)에 있다. **신규 다운로드는 0 GB다**(로컬 미러 사용). 아래 저장량은 G1 `assets/`에 생길 파생물이다.

| 단계 | 내용 | 저장 | 계산 |
|---|---|---:|---|
| A0 | Tahoe 공식 DE에서 log2FC 행렬(유전자 5,000개 × 조건 65,218개) 추출 | 약 2.6 GB | 16 CPU로 약 10분(요약 pass 실측 9.5분). padj는 필터로 쓰지 않는다 |
| **A1 (우선)** | 로컬 샤드 3,388개를 한 번 읽어 50개 세포주 × 1,344 well의 원시 count pseudobulk와 DMSO pseudo-well을 만든다 | 압축 약 2.7 GB(dense float32 16.9 GB) | 세포 9,560만 개. SNU-423 기준 1.5M 세포/241 s로 환산하면 단일 흐름 약 4.3 CPU-h, 16 worker로 20–40분 예상(추정) |
| A2 | 세포 수준 부분집합: (세포주, well)당 최대 100개 세포 | 최대 672만 개 세포, 전 유전자 약 24 GB, HVG 2,000개면 5–8 GB(추정) | chemCPA 세포 수준 1 A100 수 시간(추정) |
| A3 | 학습 세포주 기준 선택: 비간 48개 중 10개 이상에서 strong인 drug×dose 232개 + DMSO, well당 100개 세포 | 약 7 GB | A2보다 작음 |
| 대안(로컬 미러가 없을 때) | HF 재다운로드. 세포주 부분집합도 전체 337.6 GB 스캔 필요(스트리밍 가능, SNU-423 네트워크 추출 1 h 37 min). plate 부분집합은 약 24 GB/plate, 공식 DE는 88.9 GB | — | — |

**설계 조건(누출 방지):**
1. SNU-423의 처리 well은 전부 보류한다. SNU-423 DMSO 세포를 문맥 입력으로 쓰는 경우는 별도 regime으로 표기한다.
2. 약물·유전자·하이퍼파라미터는 학습 세포주에서만 고른다. HepG2/C3A 또는 비간 세포주 하나를 validation 세포주로 쓸 수 있다. SNU-423 반응으로 조건을 고르지 않는다(위 61%는 사후 기술 통계다).
3. 필수 기준선: 무변화, 학습 세포주 평균 반응(같은 drug×dose), **같은 well의 다른 세포주 평균**, 최근접 세포주(DMSO 발현 기준). 같은 well이 공유하는 기술 효과 때문에 이 기준선들이 강하게 나올 수 있다.
4. 주지표는 검출 가능 조건(모델과 무관한 DMSO null로 미리 정의)과 전체 조건에서 따로 보고한다. 조건의 약 2/3는 DMSO well 간 차이와 구별되지 않으므로, 전체 평균 지표는 잡음이 지배한다.
5. 용량은 범주형으로만 다룬다. 용량–반응 곡선이나 외삽은 plate 교란 때문에 주장하지 않는다. drug×dose 대부분이 well 1개뿐이므로 불확실성은 drug 단위 재표집으로 보고한다.
6. 모델 후보: chemCPA, 선형·잠재 가산 기준선, State, PIE. **Tahoe-100M으로 사전학습된 공개 checkpoint(State 계열 등)는 SNU-423을 학습했을 가능성이 있어**, 미관측 세포주 주장을 하려면 학습 범위를 먼저 확인해야 한다(미확인).
7. EmeraldBay SNU-423은 다른 실험, 다른 시점(5일), 다른 용량에서 같은 약물 9개에 대해 반응 방향을 외부 확인하는 데만 쓴다.

## 8. 판정

| 자료 | 판정 | 근거 요약 |
|---|---|---|
| Tahoe-100M SNU-423 | **조건부 적격** | 실제 처리 후 scRNA와 plate-matched DMSO가 있고, 약물×용량이 1,137개, 학습 세포주 48개가 같은 조건을 공유한다. 반면 반응 검출률이 31.5%(약물 기준 67%)에 그치고, 대부분 well 1개이며, 용량–plate 교란과 well 기술 효과 공유가 있다. |
| Tahoe-100M HepG2/C3A | **조건부 적격(proxy)** | 같은 설계이지만 간모세포종이며 세포와 UMI가 더 적다. |
| Tahoe-100M 나머지 48개 세포주 | **적격(학습원)** | 로컬에 있어 다운로드가 필요 없다. 공식 DE는 null 미보정이다. |
| Tahoe EmeraldBay | **조건부** | 외부 확인용. 반복이 없고 약물 20개다. |
| GSE264667 HepG2 CRISPRi | **적격(유전적, F3)** | 간모세포종이며 이전 노출이 있다. |
| GSE238219 HepaRG CRISPRi, GSE281160, GSE291757 | **조건부** | 표적이나 약물이 적거나 라벨이 불완전하다. |
| GSE284321 / GSE296254 HTTr HepaRG | **적격 또는 조건부(bulk)** | bulk이므로 scRNA 과제와 분리한다. |
| GSE149614, GSE206325, GSE151530, GSE125449, Zenodo 간 아틀라스 | **부적격** | 통제된 개입 후 RNA 정답이 없다. |

## 9. 수행 기록

- **실측:** Slurm job 20325(메타데이터, c4, 4 CPU/48 GB, 3.5분), 20326(공식 DE 88.9 GB 요약, c3, 16 CPU/96 GB, 9.4분), 20327(SNU-423 pseudobulk, c4, 4 CPU/48 GB, 4.0분), 20328(HepG2/C3A pseudobulk, c4, 2.5분), 20329(EmeraldBay 메타데이터, c3, 2 CPU/24 GB), 20339(효과 분석 최종, c4, 4 CPU/32 GB).
  - 폐기 또는 실패: 20334(KeyError), 20335와 20336(폐기한 null 정의), 20337(srun 프로토타입, 2 CPU/16 GB).
  - sacct DB 접속 실패로 MaxRSS는 기록하지 못했다. 상세는 [`F4/f4_run_record.json`](F4/f4_run_record.json)에 있다.
- **환경:** 새 venv `/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver_feedback_20261110/envs/f4`(pyarrow 17, pandas 2.2.3, anndata 0.10.9, h5py 3.11, scipy 1.12). 기존 `.venvs/scgpt`에 pyarrow가 없어서 만들었고, 기존 venv에는 아무것도 설치하지 않았다.
- **파생 자산:** G1 `assets/f4/`(1.2 GB): well pseudobulk npz 2개, 세포 QC parquet 2개, 간 세포주 공식 DE parquet 2개.
- **미실행:** 50개 세포주 전체 pseudobulk 추출(A1), 모델 학습·예측, EmeraldBay 발현 분석, GSE238219/GSE284321 다운로드.
- **차단·미확인:** 사전학습 checkpoint의 SNU-423 노출 여부, 데이터상의 24 h 시점, MIX-Seq 간 세포주 포함 여부, 로컬 Tahoe 미러 사용에 대한 소유자 확인.
- **코드:** `src/feedback/f4_tahoe_metadata.py`, `f4_tahoe_pseudobulk.py`, `f4_tahoe_pbde.py`, `f4_tahoe_effects.py`, `f4_emeraldbay_meta.py`, `experiments/feedback/f4_tahoe.sbatch`.
