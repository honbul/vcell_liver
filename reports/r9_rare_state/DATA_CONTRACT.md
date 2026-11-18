# R9 DATA_CONTRACT — 정답 척도·독립 단위·분할·허용 문맥

## 기획서

- 실행 근거: 비공개 기획서 `VCELL_R9_DONOR_SUPPORTED_RARE_STATE_PLAN_2026-11-18.md` (356줄, 29,750 bytes, sha256 `6b58e37e1ec0b7cea4b4e503902af4ba39db6e723be731eb6cae4c8cabcbb118`). 본문은 Git에 넣지 않는다.

## 원본과 가공본

| 자료 | 경로(서버, 읽기 전용) | 크기 / sha256 | 용도 |
|---|---|---|---|
| GSE149614 count | `/mnt/S1/sdata/pubdata/Virtual_Cell/GEO/GSE149614/GSE149614_HCC.scRNAseq.S71915.count.txt.gz` | 165,349,783 B / `6010d634…ab46bf` | 개발 |
| GSE149614 metadata.updated | 같은 폴더 `GSE149614_HCC.metadata.updated.txt.gz` | 489,882 B / `4c547861…78c8` | 저자 label·환자·시료·부위 |
| GSE149614 sparse 변환본 ([G1] `scripts/01_prepare_hcc.py`) | `/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver/processed/GSE149614.h5ad` | sha256 `b234f5ca…be6d` (G1 `manifests/local_artifacts.json`와 일치 확인) | 재사용(읽기 전용) |
| GSE151530 mtx/barcodes/genes/Info/series matrix/Table S1 | `/mnt/S1/sdata/pubdata/Virtual_Cell/GEO/GSE151530/` | `manifests/r9_rare_state/a0_data_hashes.json` | A4 별도 코호트 |
| GSE151530 HCC 변환본 (R9 A0) | `assets/cache/gse151530_hcc_counts.h5ad` | `a0_data_hashes.json` | HCC 세포만, count 그대로 |

모든 원본 파일의 크기·sha256은 `manifests/r9_rare_state/a0_data_hashes.json`에 있다. 원본은 수정·재다운로드하지 않았다.

## 정답 척도

- **Tier 1 (주 과제):** GSE149614 저자 `celltype` 6개 대분류. 정답은 기존 atlas 주석이며 wet-lab 검증 정답이 아니다.
- **Tier 2 (탐색):** 저자가 배포한 `res.3` 군집 번호(Myeloid 14개, B 5개). 군집은 시험 환자 세포까지 포함해 비지도로 정의되었으므로 "군집 번호 재현"이다. 전문가 세부 상태 이름이 아니다.
- **외부(A4):** GSE151530 HCC 세포의 저자 `Type`. 대응: T cells→T/NK, B cells→B, TAMs→Myeloid, TECs→Endothelial, CAFs→Fibroblast. Malignant cells·unclassified는 대응 없음(점수 계산에서 known label로 쓰지 않음; 보류·예측 분포만 기술). Hepatocyte는 외부 정답이 없어 정의불가.
- 부모(상위 유형) 출력: Immune(B, T/NK, Myeloid), Stromal(Endothelial, Fibroblast), Parenchymal(Hepatocyte). 모델 실행 전 고정.

## 독립 단위와 분할

- 독립 단위는 **환자(donor)**. 같은 환자의 여러 시료(조직 부위 Tumor/Normal/PVTT/Lymph)는 항상 같은 분할에 둔다. GSE151530의 반복 생검(H34a/b/c 등)은 Table S1 각주로 같은 환자임을 확인하고 하나의 donor로 묶는다.
- 개발: GSE149614 10명 바깥 leave-one-patient-out(LOPO). 모델 선택이 필요한 경우 바깥 시험 donor를 쓰지 않는 source 내부 donor-group 분할(3 group)을 쓴다.
- seed·회전 반복은 같은 donor 안의 기술 반복이며, donor 안에서 평균한 뒤 donor 단위로 통계를 낸다. 신뢰구간은 donor bootstrap(10,000회) 백분위와 t-구간.
- 외부: GSE149614 10명 전체로 학습한 고정 버전을 GSE151530 HCC 25명(32 시료)에 한 번 적용.

## 허용 문맥

- 학습: source donor 세포의 count, 저자 label, donor ID(가중치·분할·지지도 계산용).
- 추론 입력: 시험 세포의 count와 유전자 이름만. 시험 donor의 label, donor별 학습 embedding은 쓰지 않는다. 예외: scPoli는 공식 query surgery 단계에서 시험 donor RNA로 비지도 적응 → **transductive** 조건으로 따로 표시.
- 표현 적합(HVG/scaler/PCA)은 학습 donor 세포만 사용. HVG 후보 유전자 목록은 두 코호트에 모두 유일하게 존재하는 symbol로 제한(유전자 이름만 사용).
- N×D 진단의 표현은 대상 label 세포를 전혀 쓰지 않고 적합(`excl_t`) → label과 RNA 정보를 함께 줄인 실험.

## 노출 이력 (`cohort_overlap.tsv`)

- HCC02·HCC09: G1 고정 시험 환자로 반복 채점 → 봉인 아님. R9의 GSE149614 결과는 전부 개발 증거.
- GSE151530 HCC: `research/phase2-v2` E00 외부 평가에서 대분류 예측이 채점됨(2026-10-26) → 이 프로젝트 기준 대분류 분류의 blind cohort가 아니다. R9 A4는 "R9 고정 버전의 첫 외부 적용"이며 새 blind test라고 부르지 않는다.
- GSE149614와 GSE151530 사이 공유 GEO 시료 없음. 생물학적 동일인 여부는 공개 ID로 증명할 수 없다.
