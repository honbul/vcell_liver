# 성인 HCC 처리 전사체의 공개 메타데이터 감사

**HUH7·Hep3B·SNU-423 독립 약물 처리 연구가 있다. 이번 조사에서 기존 VC와 여러 미측정 화합물의 성능을 비교할 만한 명부는 확보되지 않았다.** 공개 원논문과 저장소의 표본 메타데이터만 확인했다. GEO 다섯 자료에서 Series에 등록된 표본 119개를 전부 대조했고, BioStudies의 E-MTAB-7847에서는 sequencing 항목 27개를 실제 source 18개로 묶었다. 아래에 표시한 후보는 기존 노출 원장 및 점검한 G1 공유 경로의 accession 조회에 나오지 않았다. 개별 원시 표본과 이전 모델의 사전학습 이력 대조는 앞으로 확인할 범위다. [GEO 표본 원표](data/public_hcc_support/geo_sample_metadata.tsv) · [BioStudies source 집계](data/public_hcc_support/ebi_source_counts.tsv) · [입력 주소와 SHA-256](data/public_hcc_support/receipt.json)

| 연구 | 어느 반응을 실측했나 | 확인된 표본 지원과 적용 범위 |
|---|---|---|
| GSE151412 | HUH7·Hep3B, 24 h RNA-seq. Sorafenib, GW3965, 두 약물 조합 및 DMSO | 총 70등록 표본. HUH7 DMSO·GW3965·sorafenib·조합이 각각 8·8·9·9, Hep3B는 각각 9다. 단일 약물은 두 개며 공개 표본 명부에서 독립 배양·물리 plate와 위치를 확인하지 못했다. HUH7은 성인 HCC이며 Hep3B는 소아 HCC 유래다. [GEO Series](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE151412) · [Hep3B 등록번호](https://www.cellosaurus.org/CVCL_0326) |
| E-MTAB-7847 | HUH7·THLE5B, 24 h RNA-seq. AD80 0.05 µM, sorafenib 5 µM 및 vehicle | HUH7은 조건별 독립 biological replicate 3개가 원논문에 명시됐다. THLE5B 조건마다 등록된 sequencing 항목 6개는 source 3개이며 별도 생물 반복으로 추가 계수하지 않는다. 성인 HCC 단일 약물 2개와 정상 대조 맥락의 기술 비교다. [원논문](https://aacrjournals.org/mct/article/18/9/1506/274247/Phenotype-Based-Screens-with-Conformation-Specific) · [공식 실험 설계](https://www.ebi.ac.uk/biostudies/files/E-MTAB-7847/E-MTAB-7847.sdrf.txt) |
| GSE242333 | HUH7 모세포와 sorafenib 내성 파생주, regorafenib 5 µM 급성 24 h RNA-seq | 4조건에 독립 표본 3개씩. 내성주는 별도 세포 맥락이며 장기 적응 상태와 급성 약물 효과를 분리해야 한다. 단일 약물 기술 확인이다. [원논문](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0301663) |
| GSE242340 | HUH7, molidustat 25 µM 및 DMSO, 24/48 h 3′ RNA-seq | 시간·처리별 biological replicate 3개. 하나의 약물과 24/48 h에서 답할 수 있는 기술 질문이다. [원논문](https://link.springer.com/article/10.1007/s00018-024-05361-6) |
| GSE186280 | SNU-423·HEPG2, sorafenib 10 µM, 12 h RNA-seq | SNU-423 처리·대조 각 2표본, HEPG2 각 3표본. 기존 Tahoe의 24 h·5 µM와 다른 조건에서 SNU-423 한 약물의 기술 자료다. [GEO Series](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186280) · [처리 시간·용량](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM5643053) |
| GSE291833 | Series는 PB56874852 4·10 µM의 HUH7 세포주 RNA와 별도 xenograft RNA를 기술한다. | 15표본 중 제목상 세포주 처리·대조 9개와 xenograft 6개다. 15개 모두 GEO source에는 xenograft로 적혀 있고, 세포주로 제목이 붙은 개별 표본에도 동물 투여 프로토콜이 붙어 있다. 원자료 제공자의 실험 명부로 이 불일치가 해소돼야 사용 가능하다. [GEO Series](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE291833) · [예시 표본](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM9378793) |

판정에 가장 가까운 두 연구인 GSE151412와 E-MTAB-7847을 합쳐도 이름으로 구분한 단일 약물은 세 개다. Sorafenib이 두 연구에 공통이며, 동일 용량과 독립 plate까지 맞는지는 별도다. 세포주별 표본 수나 여러 용량의 반복을 화합물 수로 더하면 새 약물 일반화의 분모가 부풀려진다. 이 자료들이 입증할 수 있는 범위는 해당 처리에 대한 제한된 방향 또는 순위 비교다.

## 큰 외부 자원의 역할

- **Chem-PerturBridge**는 표준화·매칭을 제공하는 통합판이다. LINCS Phase I/II와 Tahoe 원 실험을 포함하며 이 자료가 별도 독립 측정을 추가하지 않는다. 표본 표준화에 쓸 수 있고, 새 확인에는 원본 accession과 물리 측정의 중복 차단이 필요하다. GDPx2는 배포자가 밝힌 비간계열 네 맥락이다. [원 논문 표 1](https://arxiv.org/html/2605.31522v1#S3.T1) · [처리 코드와 재배포 조건](https://github.com/theislab/Chem-PerturBridge) · [GDPx2 공식 명부](https://datapoints.ginkgo.bio/updates/gdpx2-dataset)
- **CIGS·VCPI**의 대규모 측정은 각각 MDA-MB-231·HEK293T와 THP-1이다. CIGS 논문에 별도 HEPG2 JQ1 예시가 있지만 대규모 HCC 처리를 뜻하지 않는다. 이 자료의 적합한 역할은 별도로 감사한 source 입력이다. [CIGS 원논문](https://www.nature.com/articles/s41592-025-02781-5) · [VCPI 공식 명부](https://thevirtualcell.com/)
- **DILImap**의 300개 약물은 24 h 일차 간세포 반응이다. 공개 GEO의 count export와 별도 접근 정책이 적용되는 원고용 자료 범위를 구분해야 한다. 성인 HCC 처리 정답과는 세포 맥락이 다르다. [공식 GEO 등록](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM9248305) · [원논문과 접근 조건](https://pmc.ncbi.nlm.nih.gov/articles/PMC12615732/)

기존 R18의 HUH7 C3는 기술 확인 규모이며, C1은 HEPG2와 이전 채점 노출을 포함한다. 별도의 성인 HCC 우위를 주장하려면 다양한 화합물을 여러 독립 HCC 맥락에서 동일 시간·농도로 측정한 자료가 필요하다. 최소 확인 명부에는 무작위화한 물리 위치, 독립 배양·plate, 동일 assay의 vehicle 역할, 공통 유전자와 기존 VC가 사용할 입력 수준을 함께 올려야 한다. [기존 HCC 지원 감사](../r17_lincs/INVENTORY.md) · [R18 입력 명세](R18_CONFIRMATION_PROTOCOL.md)

후속 감사에서 24 h·5 µM으로 문서가 일치하는 세 약물을 분리했다. GSE151412는 GEO와 논문의 농도 표기가 달라 정확 용량 그룹에서 보류했다. [정확 조건 외부 명부](HCC_EXTERNAL_PREFLIGHT.md)

별도 공개 GEO 10개 연구의 모든 표본 150개를 추가 대조했다. SNU-423과 SNU-387의 공통 72 h vorinostat은 각 맥락에서 처리·차량 1개씩이었다. [추가 외부 지원](HCC_EXTERNAL_FOLLOWUP.md)
