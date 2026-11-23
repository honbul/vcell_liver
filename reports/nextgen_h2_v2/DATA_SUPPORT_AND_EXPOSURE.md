# H2-v2 입력 지원과 노출

## 고정 분모

H2-v1과 동일한 1,186 exact parents·1,001 대표 골격·670,830 native 관측과 1,014개 기저 RNA 세포를 사용한다. 이전에 봉인한 네 C1/H2-v1 예측 파일의 해시·행 순서·예측값을 최종 채점에서 각각 대조한다. Outer scaffold와 nested inner fold를 재분할하지 않았다. GDSC1은 신규 66골격·713세포·39,545관측의 v1 보류 명부다. GDSC1과 DRP는 세포를 공유한다. LIMORE 50성인 계통·68약물과 LICOB 38성인 공여자·76약물 및 소스 미포함 18약물은 이전 결과 노출 자료다. [공통 분할과 노출](../nextgen_h2_multiview/DATA_SUPPORT_AND_EXPOSURE.md) · [정답 없이 확인한 네 로스터](v1_registration_replay.json)

## 약물 뷰

| 입력 | DRP exact 부모 | DRP 대표 골격 | 측정 축 |
|---|---:|---:|---|
| LINCS 24시간 평균 | 724 | 619 | 978 / 10,174 / 12,328 |
| LINCS 평균 + 세포주 SD/IQR | 709 | 606 | 각 축 × 2 |
| LINCS 6/24시간 공통 | 484 | 415 | 각 축 × 2 |
| LINCS 24시간 중/고농도 공통 | 635 | 552 | 각 축 × 2 |
| Cell Painting cpg0004 | 411 | 362 | 배포 캐시 2,198열, 적합 내 관측 축 선별 |
| Tahoe 평균 / 이질성 / 용량 | 57 | 56 | 965 / 2,895 / 1,930 |

LINCS는 v1에서 잠근 비간계열 147개 참조 세포의 Level 5 서명 82,740개를 그대로 추출했다. 같은 부모·세포·시간·농도의 판/서명은 직접 측정 landmark Spearman 가중 MODZ로 결합했다. 전체 축에는 BING 외 추정 유전자까지 포함한다. landmark 24 h 지원은 v1과 동일하며 확장 추출의 합산 순서에 따른 최대 절대 차이는 1.963e−6이다. v1 대조군은 이전에 봉인한 실제 예측을 사용한다. [LINCS 확장 영수증](lincs_assets_receipt.json) · [연구 원문](https://doi.org/10.1016/j.cell.2017.10.049)

Tahoe 카탈로그 379행에서 구조가 해소된 352 exact parents를 얻었다. 등록 DRP와 지정 농도를 동시에 만족한 부모는 57개이며, 기술 외부 로스터까지 합치면 62개다. 연결성 또는 이름만 일치한 약물은 exact 집합에 넣지 않았다. 배포 raw Parquet 3,388개를 순회했다. HepG2/C3A, SK-HEP-1, SNU-423을 제외하고, 고정 바코드 해시 6%와 count QC를 적용한 1,068,266개 단일세포를 집계했다. 45개 비간계열 참조 세포주에서 plate·세포 동일 DMSO와 매칭한 처리 웰이 8,978개였다. 평균은 합산 카운트의 log1p CP10k 대비다. 이질성은 처리 평균의 세포주 간 SD 및 세포 내 log1p CP10k SD의 DMSO 대비를 포함한다. 기울기는 동일 세포 내 0.05/0.5/5 µM 중 둘 이상에서 log 농도에 적합한 변화량이다. [Tahoe 등록](tahoe_registration_receipt.json) · [실측 집계](tahoe_assets_receipt.json) · [공개 자료 설명](https://huggingface.co/datasets/tahoebio/Tahoe-100M)

Cell Painting은 v1 A549 48 h 캐시와 동일하다. 독립 가중치만 inner-fold에서 선정하며, PCA·whitening·sphering·고분산 비중복 선택을 학습 약물에서 적합한다. [Cell Painting 캐시 등록](../nextgen_h2_multiview/DATA_SUPPORT_AND_EXPOSURE.md)

## 세포 뷰

| DepMap 25Q3 실측 | 등록 DRP 세포 / 1,014 | GDSC1 세포 / 713 | 원래 특징 수 |
|---|---:|---:|---:|
| Hotspot mutations | 1,005 | 712 | 537 |
| Gene CNV | 707 | 504 | 19,366 |
| Fusions | 1,014 | 713 | 35,716 |
| Global signatures | 800 | 559 | 6 |
| Chronos CRISPR 참고 | 754 | 540 | 18,435 |

ModelID와 default ModelCondition으로 매칭했다. 융합은 공개 융합 파일에서 default RNA sequencing identity가 확인된 모델로 제한했다. LIMORE·LICOB 쿼리에는 DepMap 직접 측정 행이 없다. RNA-only 비교는 v1 RNA와 그 RNA로부터 추정한 보조 특성을 그대로 보존한다. 직접 측정한 Chronos 파일의 추가 arm은 참고 비교에 한정된다. 표적항암 관련 기술 분석은 등록 ChEMBL 집합의 사전 고정 암 관련 유전자 표적 주석을 기준으로 하며, 약물의 임상 적응증을 정의하지 않는다. [DepMap 영수증](omics_assets_receipt.json) · [DepMap 데이터 정의](https://depmap.org/portal/data_page/)
