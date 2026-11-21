# R14 자료 준비 결과

G1의 `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r14_cross_organ/raw`에 GEO 및 Figshare 원본 네 묶음, 총 **6,149,542,023바이트**를 확보했다. 기존 공유 GSE296254는 읽기 전용으로 참조했다. 수집은 SHA-256을 기록했고, 초기 감사에서는 처리 발현 수치를 읽거나 모델을 학습하지 않았다. [실험 계획](PLAN.md) · [검증 요약](data/summary.json)

| 자료 | 실측 준비 상태 | 영수증 |
|---|---|---|
| Sci-Plex3 | 3.765 GB. 원본 카운트와 바코드·유전자·표본 메타데이터를 과거 GEO 다운로드 해시와 대조했다. 메타데이터에서 24시간 웰 4,608개와 A549 72시간 웰 384개를 구분했다. 24시간은 세포주당 1,536웰, vehicle 32웰, 양 반복에 있는 약물·용량 752조건이다. | [GEO 수집](data/sciplex3_receipt.json) |
| MIX-seq | 1.138 GB. Figshare release 3의 24시간 experiment 3과 experiment 10 원본 ZIP 14개, 설계표 및 설명서를 확보했다. 출판자 파일 MD5를 대조했다. | [Figshare 수집](data/mix_seq_receipt.json) |
| HepaRG GSE284321 | 0.719 GB. 29개 count 파일의 저자 MD5가 모두 일치했다. 표본 메타데이터와 count 열 32,886개가 모두 매칭됐다. 24시간 화학물질명 1,201개를 확인했다. | [GEO 수집](data/heparg_bulk_receipt.json) |
| HepaRG GSE296254 | 공유본. 행렬과 메타데이터 표본 ID 8,598개가 모두 매칭됐다. 24시간 화학물질명 317개와 vehicle 행 118개를 확인했다. | [검증 요약](data/summary.json) |
| HepaRG GSE299113 | 0.528 GB. 공개 H5AD와 GEO SOFT를 확보했다. 표본 478개이고, 두 배치 중 EPA0001에만 DMSO 60개가 기록됐다. 배치 내 대조가 필요한 분석은 처리와 vehicle이 함께 있는 조건으로 제한한다. | [GEO 수집](data/heparg_single_cell_receipt.json) |

표본 인덱스는 G1 `metadata/v4`에 보관했다. 처리 행별 물리 plate와 대조군, Sci-Plex3 반복 ID, MIX-seq의 세포주와 ZIP 내부 파일을 기록했다. GSE284321의 첫 plate group을 GSE296254 공개 probe 주석과 정확한 ID로 연결하니 16,632개 probe에서 Tahoe의 고유 Ensembl 유전자 15,125개로 연결됐다. 이후 모든 plate group의 probe 집합과 실제 평가 패널의 교집합을 확인한다. 화학물질명과 설계 웰 수는 화학군 중복 제거 및 발현 QC 전 값이다.

## 간암으로 검증할 수 있는 외부 단일세포 표본

MIX-seq 설계표와 원본 ZIP의 `classifications.csv`를 함께 봤다. 원본 분류에서 해당 세포주에 배정되고 저자가 `normal`로 분류한 세포가 대조군과 모든 계획 처리 조건에 존재했다. 보충표의 `DepMap ID` 열은 네 세포주 모두 ZIP 내부 ID와 달랐다. 세포주 이름과 ZIP의 분류 ID를 Cellosaurus에 대조해 연결한다. [원본 분류 확인표](data/mix_seq_liver_assignments.tsv) · [Hep 3B2.1-7](https://www.cellosaurus.org/CVCL_0326) · [JHH-5](https://www.cellosaurus.org/CVCL_0364) · [Li-7](https://www.cellosaurus.org/CVCL_3840) · [SNU-761](https://www.cellosaurus.org/CVCL_5089)

| 원본 배정 | 실험 | 24시간 약물 | DMSO 세포 | 약물 조건별 세포 범위 |
|---|---:|---:|---:|---:|
| Hep 3B2.1-7 `ACH-000625` | 3 | 4 | 23 | 22–51 |
| JHH-5 `ACH-000734` | 3 | 4 | 45 | 51–71 |
| Li-7 `ACH-000471` | 3 | 4 | 42 | 49–73 |
| SNU-761 `ACH-000537` | 10 | 8 | 54 | 15–48 |

표의 세포 수는 물리 반복 수와 별개다. 두 실험에 공통인 Trametinib을 한 약물로 계산하면 11개 약물이다. 최종 검증에는 용량, 실제 대조군, 독립 처리 웰과 세포 QC를 고정한 표본만 쓴다.

## Sci-Plex3를 추가했을 때 화학 구조 범위

24시간 약물 카탈로그 188개에서 모두 SMILES를 해석했다. 그중 25개 항목은 Tahoe 카탈로그와 동일한 비입체 표준 모화합물 구조였다. Tahoe 카탈로그 전체와 일치한 구조를 제거한 후 Sci-Plex3의 서로 다른 모화합물 160개가 남았다. 이 구조들만 추가한 예비 ECFP 거리 검사에서 Tahoe 평가 203개 군 중 **94개 군의 가장 가까운 이웃이 더 가까워졌다.** 같은 계산에서 최대 Tanimoto가 0.25 미만인 군은 **115개에서 80개**로 줄었다. 이 숫자는 구조 범위의 변화이며, 발현 예측 이득은 실험에서 확인한다. 모화합물 구조 기반 결과에 이름·CAS·CID·tautomer 감사와 발현 QC를 추가할 예정이다. [구조 검사와 입력 해시](data/chemistry_preview.json)

## 실행 기록

Slurm 수집은 28265·28266·28268·28273에서 완료했다. MIX-seq 초기 두 제출 28267·28272는 Figshare 버전 API의 URL 필드 이름에서 실패했고, 필드 이름을 실제 응답에 맞춰 다시 수행했다. 최종 메타데이터 검사 28281은 공유 GSE296254의 유전자 주석 열을 표본에서 구분하고, 원본 ZIP의 세포주 배정 및 bulk probe 교집합을 기록했다. [수집 코드](../../src/r14_cross_organ/acquire.py) · [메타데이터 검증 코드](../../src/r14_cross_organ/audit_metadata.py)
