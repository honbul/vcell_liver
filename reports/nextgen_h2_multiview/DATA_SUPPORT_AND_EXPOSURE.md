# 데이터 지원과 평가 노출

| 등록 지원 | Exact parents | 대표 Murcko 골격 |
|---|---:|---:|
| GDSC2 · CTRPv2 · PRISM20Q2 전체 | 1,186 | 1,001 |
| Cell Painting cpg0004 | 411 | 362 |
| LINCS 비간계열 Level 5 전체 시간·농도 | 742 | 635 |
| LINCS 24 h, 0.5–20 µM | 724 | 619 |
| LINCS 6/24 h, 0.5–20 µM | 734 | 627 |
| LINCS strict QC, 최소 비간계열 1개 | 695 | 591 |
| LINCS strict QC, 최소 비간계열 2개 | 651 | 559 |
| Cell Painting과 LINCS 6/24 h 동시 보유 | 378 | 332 |

등록 계획의 405/796은 초기 교집합 추정이다. 고정한 1,186 exact-parent 명부에서 Cell Painting이 411개로 확인됐다. LINCS는 공표 raw InChIKey의 일치 819개와, 구조를 정규화하고 pert ID 충돌을 제거한 실제 exact 구조·서명 일치 744개를 구분한다. 알려진 비간계열의 서명 742개 중 6/24 h와 0.5–20 µM 조건에서 734개를 사용한다. 데이터 영수증의 LINCS 796은 DRP와 기술 검증 후보를 합친 조회 우주 전체의 비간계열 Level 5 일치 수다. 670,830행은 스크린별 CTRPv2 285,842, GDSC2 83,995, PRISM20Q2 300,993개다. outer 평가 부모는 396 / 395 / 395개다. 세 outer 및 9개 inner 검증 분할과 OOF 하위 적합의 exact parent·connectivity·모든 scaffold 표현 겹침은 0개다. [폴드별 지원](STEP0_SUPPORT.tsv) · [데이터 영수증](step0_receipt.json) · [분할 검사](split_preflight.json)

## Cell Painting

고정 캐시는 A549 48 h, dose recode 6에서 배포된 DMSO 기준 중앙값으로 구성된다. 2016년 batch 1의 해당 행 1,558개 중 1,547개, 2017년 batch 2의 360개 중 353개에 정확 구조가 연결됐다. 두 배치를 캐시 생성 때 부모별 중앙값으로 접었다. 캐시에는 판·웰·개별 반복과 per-well QC가 남아 있지 않아 재추정 가능한 범위는 fold별 형태 특징 변환까지다. 동일 411개 중 130개는 2,198개 축, 262개는 1,781개 축, 19개는 1,745개 축이 있었다. 적합에서 완전히 관측된 특징만 사용하고, 인정한 축의 드문 query 공란에는 해당 fit 평균을 적용한다. 일부 캐시 값의 크기가 1e19에 달해 평균·분산 계산에는 float64를 썼다. [캐시 분석 코드](../../src/nextgen_h2/assets.py) · [공개 데이터 설명](https://github.com/broadinstitute/cellpainting-gallery)

## LINCS 2020

배포 Level 5의 chemical perturbation만 사용했다. `cellinfo_beta.txt`에서 조직이 알려진 비간계열을 허용했다. HEPG2, HUH7, JHH5, JHH7 및 미확정 계열 82개를 제외하고 예약 HCC 코호트 별칭과 Cellosaurus 식별자를 별도로 대조했다. 등록 조회 우주 및 6/24 h window에 147개 참조 세포주, 82,740개 서명, 4,799개 plate가 있다. 서명 추출에는 bead batch, project, plate, well, replicate ID, replicate 수, cc_q75, QC pass와 농도·시간을 보존했다. 비간계열 중 일부는 반응 패널과 공통 세포주여서 assay-assisted 평가에는 서로 다른 assay의 측정 정보가 함께 쓰인다. 알려진 비간계열 참조 중 122개는 반응 source의 DepMap RRID에도 존재했다. 표적 효능 AUC, 간계열 LINCS 서명 또는 외부 반응값을 뷰 생성에 사용한 적은 없다. [메타데이터 해시와 반복 분포](step0_receipt.json) · [CMap 배포 설명](https://clue.io/connectopedia/guide_to_geo_l1000_data)

배포 Level 5는 이미 Broad의 생물 반복 MODZ다. 이 연구의 추가 MODZ는 독립 Level 5 서명 집합의 동일 약물·세포·농도·시간 내 합의이며, 978 landmark에 대한 서명 간 Spearman 가중합이다. 음의 상관을 .01로 제한하고 대각을 0으로 하며, 농도·시간·세포에 차례로 동일 가중을 준다. strict QC는 cc_q75≥.20, 원 반복≥2, publisher QC pass를 요구한다. 경로 arm은 직접 측정 및 best-inferred BING의 PROGENy 14개와 Hallmark 30개를 사용한다. 978 landmark arm에는 inferred 값이 포함되지 않는다. [공식 MODZ 방법](https://escholarship.org/uc/item/6t74p3f0)

## 기술 외부 검증과 잠금

- LIMORE 기존 성인 HCC 50계통·68약물과 LICOB 43 PDO·38성인 공여자·76약물은 이전 결과 노출 코호트다. LICOB exact source-absent 19개 중 입체가 완전히 해소된 18개를 별도 보고한다. LINCS와 Painting의 지원은 각각 LIMORE 59/44, LICOB 55/35, LICOB18 9/3이다. [기존 LIMORE 사용 이력](../vc_build_v1/v0plus/LIMORE_CORRECTED_RESULTS.md) · [LICOB 등록 및 노출](../vc_build_v1/v0plus/LICOB_COLD_RESULTS.md)
- GDSC1 8.5 공개 원본은 기존 G1 캐시에서 해시를 확인했다. 402개 native ID 중 IMPROVE 구조와 공식 명칭이 맞고 입체가 해소된 후보를 찾은 뒤 학습에 있는 exact parent, connectivity, 모든 scaffold 표현 및 이전 GDSC1 브리지의 약물/ID를 제외했다. H2에서 66개 독립 골격, 67 native ID, 713 cell과 39,545 cell-parent 쌍을 AUC 열 없이 등록했다. CP 13개와 LINCS 6/24 h 48개가 관측 돼 있다. 같은 공개 릴리스의 선행 실험은 주로 개발 코어 약물을 다뤘으며 H2와 세포가 공유된다. H2가 새로 고정한 골격 보류 결과이며 연구 전체 수준의 미접촉 또는 환자 독립성을 주장할 근거는 없다. 그 밖의 gCSI는 같은 수준의 구조·기저 RNA·반응 키 등록 없이 추가하지 않았다. [기존 브리지](../vc_build_v1/v0plus/GDSC1_BRIDGE_RESULTS.md) · [공식 GDSC 원천](https://depmap.sanger.ac.uk/documentation/datasets/drug-sensitivity/)
- 원본과 큰 계산물을 G1에 둔다. 이 작업은 G1에 이미 등록된 공개 원자료·캐시를 읽기 전용으로 사용했다. [실행과 원본 해시](step0_receipt.json)
