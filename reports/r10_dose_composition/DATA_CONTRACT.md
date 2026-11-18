# R10 DATA_CONTRACT

## 1. 원본 자료 (GEO, 공개)

- Series GSE139944, sample GSM4150378 (sci-Plex3: A549·K562·MCF7, 188 화합물 × 10 nM·100 nM·1 µM·10 µM, vehicle).
- 받은 경로: NCBI GEO 공개 HTTPS(`ftp.ncbi.nlm.nih.gov/geo/samples/GSM4150nnn/GSM4150378/suppl/`), 일반 User-Agent, 인증 없음. 접근 시각 2026-11-18T09:11:55Z(UTC). 크기는 GEO `filelist.txt`와 일치, gzip 무결성 확인, sha256을 job 26223이 기록하고 별도 job(`r10_verify`)에서 `sha256sum -c`로 재확인했다.
- 전체 receipt: [`manifests/r10_dose_composition/raw_data_receipt.json`](../../manifests/r10_dose_composition/raw_data_receipt.json). 서버 위치 `assets/raw/GSE139944_GSM4150378/`(Git 제외).

| 파일 | bytes | sha256 (앞 16자) | 사용 |
|---|---:|---|---|
| `GSM4150378_sciPlex3_pData.txt.gz` | 84,673,357 | `860ba5c21846e2cf` | 세포 메타데이터·hash QC·용량·replicate·plate |
| `..._screen_cell.annotations.txt.gz` | 2,770,264 | `5db11591747a49f0` | 행렬 열 순서(pData와 순서 일치 확인) |
| `..._screen_gene.annotations.txt.gz` | 806,168 | `fbe43028cfb75dc5` | 유전자(사람 ENSG 58,347개만 사용) |
| `..._hashTable_metadata.txt.gz` | 64,715 | `98a65191d343d149` | well 목록(세포 0인 well 포함)·SMILES |
| `GSM4150378_sciPlex3_hashSampleSheet.txt.gz` | 6,441 | `ac66d0dfdcc3c170` | 보관만 |
| `..._screen_hashTable.out.txt.gz` | 580,400,009 | `91327287c19e6bca` | 보관만(hash read 원표; pData 요약을 사용) |
| `..._screen_UMI.count.matrix.gz` | 3,096,224,606 | `7d632716aa6ed0fc` | UMI triplet(유전자, 세포, count), 1,007,419,688행 |

받지 않은 파일: 세포주별 24 h RDS, `cds_all_cells.RDS`, 다른 sci-Plex sample. 가공 배포본(scPerturb/pertpy)은 **받지 않았고 교차확인도 하지 않았다**(미실행; §계획 4의 선택 항목).

## 2. 단위와 QC

- 세포 → culture well(hash combo = well oligo × plate oligo) → (세포주, replicate, 약물, 용량) → 약물.
- pData의 `culture_plate`는 plate hash oligo와 모든 well에서 일치했다. 각 replicate는 서로 다른 plate 8장(A549 24 h는 rep1 plate 9–16, rep2 33–40; K562 1–8/25–32; MCF7 17–24/41–48)에 놓이고 plate마다 vehicle well 2개가 있다. 같은 약물의 네 용량이 같은 plate인지 여부는 `eligible_triplets.tsv`의 `endpoints_same_plate`, `query_same_plate_as_endpoints`에 있다.
- 두 replicate는 서로 다른 plate 묶음이지만, 메타데이터만으로 "독립 배양 batch"인지 확정할 수 없다. 따라서 cross-replicate 결과를 "batch 없는 생물학적 전이"로 해석하지 않는다. vehicle은 replicate별로 16 well을 합쳐 사용한다(같은 plate 대조 매칭은 하지 않음).
- sequencing 단위: PCR plate 2개(E09, F10), RT plate 4, ligation plate 4 — well과 교차 배치되어 있다(분석에서 보정하지 않음).
- hash QC(R10 사전 고정 규칙): `qval_W<0.05`, `qval_P<0.05`, top-to-second-best ratio W·P ≥ 10, n.umi ≥ 500, 24 h. 원논문의 정확한 threshold와 같은지는 **미검증**이다.
- 결과: pData 799,317 → hash 배정 762,795 → 24 h 680,685 → QC 통과 497,356 (A549 122,420 / K562 134,025 / MCF7 240,911). 세포 수는 생존율의 정답으로 쓰지 않는다.
- 세포 단위 barcode 맵은 서버 `assets/cache/r10_v1/cells_all.parquet`(Git 제외; sha256은 `manifests/r10_dose_composition/server_artifacts.tsv`). well 단위 맵은 [`dose_replicate_map.tsv`](dose_replicate_map.tsv)(4,608 well; 세포 0 well 포함).

## 3. 분할 (약물 단위)

- [`manifests/r10_dose_composition/drug_split.tsv`](../../manifests/r10_dose_composition/drug_split.tsv): train 115 / development 35 / test 38 약물. pathway_level_1 층화, 동일 SMILES 묶음(해당 쌍 0개), seed 20261118. 구조 scaffold 묶음은 하지 않았다(구조 toolkit 미사용).
- 같은 약물의 모든 세포주·용량·replicate는 같은 역할. 세 세포주 모두 학습에 등장한다(세포주 보류 없음).
- PCA·HVG·표준화·상태 atlas·head 가중치·특징 표준화는 train 약물(+vehicle)로만 적합. development 약물은 선택에만, test 약물은 freeze 후 한 번 평가.

## 4. 과제와 허용 정보

| episode | endpoint (입력) | query (정답) | t |
|---|---|---|---|
| E100 | 10 nM, 1 µM | 100 nM | 0.5 |
| E1000 (주) | 100 nM, 10 µM | 1 µM | 0.5 |

- 입력: 같은 약물·세포주의 endpoint 두 well의 세포(PCA 좌표·고정 membership), endpoint replicate의 vehicle 세포, 요청 용량(episode), 세포주.
- 금지: query well의 RNA·평균·구성비·세포 수·depth, 다른 세포주의 같은 약물 query, 외부 약물 정보. 코드에서 예측 함수는 `Context`(query 없음)만 받는다. query n은 채점기(관측 표본 크기)와 적격성에만 쓰인다.
- 주 조건 cross: endpoint는 rep a, query는 rep b(a≠b). 두 방향은 약물 안에서 평균한다. same(a=b)은 별도 표.
- 적격성: endpoint·query well QC 세포 ≥ 20, vehicle ≥ 50. 제외 이유는 `eligible_triplets.tsv`.

## 5. 정답 척도와 지표

- 표현: train-only PCA 20 (표준화 HVG 2,000의 log1p CP10k). 이 20차원은 표준화 HVG 분산의 3.8–4.6%만 설명한다(희소한 단일세포 자료). 모든 분포 지표는 이 공간의 지표다.
- 상태: 세포주별 GMM K=4(사전 규칙으로 선택), soft membership s_k(z). 생성·조립된 구름은 같은 atlas로 다시 읽는다.
- 구성비 TV(주), JS, Brier; 상태 내부 조건부 평균 RMSE·조건부 ED(query n_eff ≥ 5 상태, query 비중 가중); 전체 ED·MMD; 유전자 Δmean RMSE(HVG log1p CP10k, 선형 PCA decoder로 모든 방법 동일 복원; decoder 재구성 오차 별도 열).
- 관측 한계: `oracle_replicate`(query 용량의 반대 replicate well) — 진단 전용, 성능표 제외.
- 집계: seed → 방향 → 세포주 동일 가중 → 약물; 불확실성은 약물 단위 paired bootstrap. 이 구간은 관측된 두 replicate·plate 환경에 조건부인 약물 간 불확실성이다.
