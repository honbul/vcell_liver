# 연구 A 데이터 카드 (단계 A0)

작성일: 2026-11-16. 이 문서는 실제로 확보하고 연결한 파일만 적습니다. 논문에 적힌 규모를 그대로 옮긴 숫자는 없습니다.
모든 숫자는 `manifests/pathway_sensitivity_a1/data_card.json`(빌드 스크립트가 직접 계산)에서 가져왔습니다.

## 1. 접근 상태

| 자료 | 상태 | 경로 |
|---|---|---|
| LIMORE 약물 반응표 | **확보됨** (서버 공유 영역에 2021-11-01부터 보관된 사본) | `/mnt/S1/sdata/pubdata/LIMORE/drug.txt` |
| LIMORE 모델 주석 | 확보됨 (같은 보관 사본) | `/mnt/S1/sdata/pubdata/LIMORE/81clinic_information.txt` |
| LIMORE RNA | 확보됨 (GEO 공식 등록 파일을 2026-11-16에 내려받음) | `assets/raw/GSE97098/GSE97098_HCC81.genes.fpkm_table.log2.txt.gz` |
| LCCL RNA·AUC·GI50·주석·약물표 | 확보됨 (원 연구실 사이트에서 2026-11-16에 내려받음) | `assets/raw/LCCL/` |
| MSigDB Hallmark | 확보됨 (서버에 이미 있던 2026.1.Hs 사본, CC BY 4.0로 기록됨) | `/mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/` |
| Cellosaurus | 확보됨 (공개 API, release 57.0) | `manifests/pathway_sensitivity_a1/cellosaurus_lookup.tsv` |

상태 판정은 `A_DATA_ACCESS_BLOCKED`가 **아닙니다**. 각 파일의 URL·받은 시각·크기·SHA-256은 `manifests/pathway_sensitivity_a1/downloads.tsv`에 있습니다.

### 접근에 실패했거나 확인하지 못한 것

- LIMORE 웹사이트(`www.picb.ac.cn/limore/`)는 연결이 거부되었고, `limore.sibcb.ac.cn`은 주소를 찾지 못했습니다.
- LIMORE 논문의 보충표(Table S1, S4)는 Europe PMC 보충파일 경로에서 받지 못했습니다(파일 없음 응답).
  따라서 서버 보관 사본 `drug.txt`가 Table S4와 같은 값인지는 **직접 대조하지 못했습니다**.
  서버 사본을 누가 어느 주소에서 받았는지에 대한 기록도 찾지 못했습니다.
- 대신 확인한 것: 서버에 보관된 RNA 사본(`HCC81.proteinGenes.fpkm_table.log2.txt`)은 GEO 공식 파일과 유전자 기호가 유일한 19,739개 유전자·81개 표본에서 값이 완전히 같았습니다(최대 절대 차이 0). 반응표의 세포 이름 81개는 RNA 열 이름·임상표 이름과 정확히 일치합니다.
- LCCL 논문 본문(구독 필요)은 열지 못했습니다. LCCL 측정 시간·농도 범위·AUC 계산식은 **확인하지 못했습니다**.
- LCCL 다운로드 페이지의 `LCCL_description` 항목은 링크가 깨져 있었으나 같은 폴더의 `LCCLdescription.txt`는 정상적으로 받았습니다. 옛 주소 `lccl.zucmanlab.com/hcc/download`는 500 오류였습니다.
- LIMORE 약물 주석표(염 형태, 공식 ID)는 확보하지 못했습니다. 약물 대응은 이름과 LCCL의 별칭 열로만 했고 염 형태는 모두 `not_verified`입니다.

## 2. 최종 표본 수

| 항목 | 수 |
|---|---|
| LIMORE 모델(RNA·반응 모두 있음) | 81 |
| 주 분석에서 제외 | 2 (HepG2, Huh6: 원 연구와 Cellosaurus 모두 간모세포종) |
| 주 분석 HCC 모델 | **79** |
| 독립 계열(`family_id`) | **76** |
| 모델이 둘인 계열 | CLC19/CLC20 (같은 종양의 subclone, 논문·Cellosaurus), HLE/HLF (같은 환자, 논문·Cellosaurus·DepMap PatientID), PLC8024/PLCPRF5 (Cellosaurus에서 같은 세포주의 다른 표기) |
| 약물 | 90 (모든 약물이 79개 모델·76개 계열에서 측정됨. 빠진 쌍 0) |
| 계열 20개 이상인 약물 | 90 (탐색용으로 분리된 약물 없음) |

- 계열은 Cellosaurus의 같은 등록번호, 조상 세포주, 같은 개인 연결, DepMap PatientID로만 묶었습니다. 이름이 비슷하다는 이유로 묶은 것은 없습니다.
- JHH1은 Cellosaurus에 "일부 JCRB 재고가 생쥐 세포로 오염" 주석이 있습니다. 사전에 정한 규칙(오분류·전체 오염만 제외)에 따라 유지하고 표시만 했습니다. LIMORE가 쓴 재고가 어느 쪽인지는 확인하지 못했습니다.
- MHCC97H는 MHCC97에서 파생되었지만 부모 세포주는 패널에 없습니다.
- 운영 기준(독립 계열 30개 이상, 약물당 20개 이상)은 충족합니다. 이 기준은 통계적으로 충분하다는 뜻이 아닙니다.

## 3. 정답(endpoint)

- 주 정답: `drug.txt`의 `Activity Area`(AA). 범위 0–1.096, 전체 쌍의 14.3%가 정확히 0입니다(예: ABT-199는 76%가 0).
- 방향: AA가 클수록 민감합니다. 근거는 자료 자체입니다. AA와 IC50의 Spearman은 −0.65, AA와 Emax는 −0.76이고, 90개 약물 모두에서 AA–Emax 상관이 음수입니다. 논문 방법에는 AA의 식이 적혀 있지 않고(GRmetrics로 계산했다고만 기술), 그림 설명에 "1−Activity Area"로 표시했다는 문장이 있습니다. 정확한 정규화 식은 **확인하지 못했습니다**.
- 반복: 논문에는 한 용량에 3개 well, 22개 모델은 2회 이상 스크린했다고 적혀 있으나, 확보한 표에는 모델×약물당 값이 하나뿐입니다(저자가 이미 집계). 여기서 추가로 평균한 것은 없습니다.
- IC50은 최대 농도에서 잘린 값이 많습니다(20이 2,317행). 이번 연구에서 IC50과 Emax는 방향 확인에만 썼습니다.
- 측정 조건(논문 방법): CellTiter-Glo, 72시간, 3배 희석 7용량 또는 2배 희석 10용량.

## 4. RNA

- LIMORE: GEO 파일은 이미 log2 변환된 FPKM입니다(최솟값 0, 최댓값 19.34). 다시 로그를 취하지 않았습니다. 논문은 pseudocount를 언급하지 않아 정확한 변환식(`log2(FPKM+1)` 여부)은 확인하지 못했습니다.
- LCCL: 원 FPKM(최댓값 8,223)에 `log2(FPKM+1)`을 한 번 적용했습니다(변환 후 최댓값 13.0). vsn 표는 쓰지 않았습니다.
- 공통 유전자 축: GEO 파일에서 protein_coding으로 표시된 19,783개 중 LCCL 표에도 Ensembl ID가 있는 19,025개, 같은 기호가 겹치는 경우를 정리해 **19,018개**입니다. 발현값이나 반응값으로 유전자를 고르지 않았습니다. 목록은 `manifests/pathway_sensitivity_a1/gene_axis.tsv`입니다.
- 두 연구의 발현을 합쳐 정규화하지 않았습니다.

## 5. 경로

- MSigDB Human Hallmark **2026.1.Hs** 50개 집합(기획서가 언급한 사이트 최신판 2026.2.Hs가 아니라 서버에 이미 있던 판입니다. 기획서가 허용한 선택입니다).
- 규칙(공통 축에서 15개 이상, 원 집합의 50% 이상)으로 제외된 집합은 없습니다. 최종 K=50, 가장 적게 남은 집합도 89.5%가 남았습니다. 집합별 숫자는 `hallmark_sets.tsv`에 있습니다.
- 점수: Bioconductor singscore 1.30.0. numpy로 옮겨 적은 계산과 두 연구 모두에서 일치했습니다(최대 차이 6e-16).
- GMT 파일 자체는 Git에 넣지 않았습니다. 경로와 SHA-256만 기록했습니다.

## 6. 외부 자료(LCCL)

| 항목 | 수 |
|---|---|
| LCCL 세포주 | 34 |
| 제외 | 2 (HepG2, Huh6) |
| LIMORE에도 있는 계열 | 27개 세포주, 26개 계열 (HLE·HLF가 한 계열) |
| LIMORE에 없는 새 계열 | 5개 세포주, 4개 계열 (HepaRG, B1, HCC-3, 그리고 같은 환자에서 나온 HCC-1.1·HCC-1.2) |
| LCCL 화합물 열 | 35 (단일 약물 31, 조합 4) |
| LIMORE와 대응된 단일 약물 | **18** |

- LCCL의 AUC는 0.14–1.16이고 GI50과의 Spearman이 +0.81이므로 AUC가 작을수록 민감하다고 판단했습니다. AUC의 계산식은 확인하지 못했습니다.
- GI50은 42%가 상한 10에 걸려 있습니다(잘린 값). 보조 지표로만 씁니다.
- B1은 LCCL 주석에 Cellosaurus 번호가 없어 이름으로 `CVCL_A1AS`에 대응시켰습니다. 이 대응이 같은 세포주인지는 확인하지 못했고, 계열 관계에는 영향이 없습니다(연결된 기록 없음).
- 새 계열이 4개뿐이므로 외부 검증 상태는 `external_validation_limited`입니다.

## 7. Git에 넣은 것과 넣지 않은 것

- 넣은 것: 코드, 설정, 모델·약물 등록표, 유전자 축 목록, Hallmark 집합 크기표, 분할표, 해시.
- 넣지 않은 것: 발현 행렬, 원 반응표(`drug.txt`, `AUC.txt`, `GI50.txt`), GMT. 이 파일들의 재배포 조건을 확인하지 못했기 때문입니다. 예측표에는 예측값만 넣고 실측값은 ID로 다시 붙이도록 했습니다.
- 복구 방법: `python src/pathway_sensitivity_a1/fetch_public.py <assets> <manifest>`가 공개 파일을 다시 받고 해시를 기록합니다. LIMORE 반응표와 임상표는 서버 공유 영역의 사본을 읽습니다.
