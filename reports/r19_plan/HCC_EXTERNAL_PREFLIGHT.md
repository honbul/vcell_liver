# 성인 간암 외부 24시간·5 µM 검사 명부

**명시적 단일 약물·시간·농도를 맞춘 기술 후보는 세 약물이다. 모두 HUH7이고 독립 배양·plate·좌표 및 vehicle 역할까지 검증된 공통 외부 코호트는 없었다.** 앞선 조사와 겹치는 후보를 포함해 GEO 13개 series의 등록 표본 263개 및 대표 GSM 9개를 대조했다. 새로 조사한 GEO 후보는 프로젝트 노출 원장의 accession 조회에 없었다. 개별 raw 표본·모델 사전학습 중복은 아직 감사하지 않았다. [기존 자료 조사](HCC_PUBLIC_SUPPORT.md) · [GEO 원표](data/hcc_external_preflight/geo_sample_metadata.tsv) · [개별 약물·농도 근거](data/hcc_external_preflight/individual_metadata.tsv) · [메타데이터 SHA-256](data/hcc_external_preflight/receipt.json)

| 자료 | 정확한 24 h·5 µM 후보 | 처리 / 대조 등록 | 다음에 확인할 조건 |
|---|---|---:|---|
| E-MTAB-7847 | HUH7 sorafenib | 3 / 3 독립 biological source | plate·좌표와 vehicle 물리 역할은 미확인이다. AD80은 0.05 µM 조건이다. [원논문](https://aacrjournals.org/mct/article/18/9/1506/274247/Phenotype-Based-Screens-with-Conformation-Specific) · [원 SDRF](https://www.ebi.ac.uk/biostudies/files/E-MTAB-7847/E-MTAB-7847.sdrf.txt) |
| GSE242333 | 모세포 HUH7 regorafenib | 3 / 3 GEO 표본 | 모세포의 급성 처리를 쓴다. 별도의 sorafenib 내성 파생 계열 6표본은 별개 맥락이다. plate·좌표의 독립성은 미확인이다. [GEO GSM](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM7758206) · [원논문](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0301663) |
| GSE271000 | HUH7 MS7126 | 2 / 2 GEO 표본 | 제목·source·처리 증거가 일치한다. 별도 HUH7 유전 개입 4표본과 HepG2 48 h 자료를 떼어 냈다. HUH7은 단일 약물 하나에 대조가 2개라 3개 vehicle 역할을 구성할 수 없다. [GEO GSM](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8366939) · [원논문](https://www.nature.com/articles/s41556-026-02048-x) |

## 용량 불일치와 다른 자료의 역할

- **GSE151412**는 HUH7·Hep3B 두 세포주의 독립 반복 8–9개를 원논문이 명시했다. GEO 표본에는 sorafenib 2 µM·GW3965 5 µM, 원논문 RNA-seq 그림 2B에는 sorafenib 5 µM·GW3965 2 µM가 적혀 있다. A 패널의 생존율 처리 조건으로 B 패널의 RNA-seq 농도를 대신 정하지 않았다. 정확한 농도 비교에는 표본별 원실험 기록이 필요하다. [GEO HUH7](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4577596) · [GEO Hep3B](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4577639) · [논문 Fig. 2B](https://pmc.ncbi.nlm.nih.gov/articles/PMC9481113/)
- **GSE96796**은 SNU-761·HUH7·Hep3B·HepG2의 sorafenib 하나를 세포주별 3 처리·3 대조 microarray로 측정했다. GEO의 3 mM와 원논문 실험 절의 3 µM가 다르다. 용량 판정 전에는 해당 측정의 sample title과 서로 다른 출처를 함께 보존한다. SuperSeries와 네 SubSeries의 동일 GSM은 한 번만 센다. [GEO GSE96794](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96794) · [원논문 실험절](https://onlinelibrary.wiley.com/doi/10.1002/hep.29237/abstract)
- **GSE284346**은 HUH7의 24 h carfilzomib 3표본과 DMSO 3표본이다. 처리 용량이 개별 명부에서 확인되지 않았고, Homo sapiens 표본의 처리 파일이 마우스 GRCm39로 정렬됐다고 기재돼 있다. 사람 유전자 축을 확인하기 전 기존 processed 파일은 사용 보류다. **GSE113005**는 HUH7 내성주의 fostamatinib 2 µM·24 h, 3 대 3으로 본 HUH7·5 µM과 구분한다. **GSE192771**의 6개 약물 RNA 표본은 논문이 sorafenib 75 µM로 설명한다. [Carfilzomib GSM](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8682719) · [Fostamatinib GSM](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM3093318) · [Sorafenib 원논문](https://www.sciencedirect.com/science/article/pii/S104366182200189X)
- **GSE309428**의 HUH7 tanshinone IIA RNA는 8 µM·48 h이고 열처리·유전 개입이 섞여 있다. **GSE185139**의 등록 표본은 untreated·GO·GO-Rg3·GO-Rg3-DOX로 자유 doxorubicin 단독 비교의 GSM이 없다. **GSE291757**에 HCC라는 제목이 붙었지만 등록 라이브러리 4개는 HepG2·48 h·배지 대조다. [GSE309428](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE309428) · [GSE185139](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE185139) · [GSE291757 원논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC13241834/)
- Alithea의 HUH7 공개 예시는 768 시료가 달린 SB431542·TGF-β 용량 및 결합 실험이다. 업체의 별도 사례 51개 화합물·6개 세포주는 배포 명부와 개별 HUH7 화합물 지원을 확인하지 못해 후보 수에 넣지 않았다. [업체 샘플 명부](https://alitheagenomics.com/kits/drug-seq-kits/) · [사례 설명](https://alitheagenomics.com/technology/drug-seq/)

## 예측 입력과 가능한 확인 범위

Sorafenib은 GSE92742의 같은 이름으로 두 연결성 키가 나온다. PubChem의 sorafenib과 일치하는 BRD-K23984367만 구조 후보로 확인했고 BRD-K49810818을 이름만으로 합치지 않는다. Regorafenib은 기존 Tahoe 구조 catalog에 있다. MS7126은 두 로컬 catalog의 정확한 이름 조회에서 매칭이 없었고 원 저자 구조 확인이 남았다. Sorafenib과 regorafenib은 서로 가까운 구조라 두 화합물로 scaffold 외삽을 검증하기 어렵다. [로컬 구조 대조](data/hcc_external_preflight/chemistry_catalog.json) · [PubChem sorafenib](https://pubchem.ncbi.nlm.nih.gov/compound/216239) · [PubChem regorafenib](https://pubchem.ncbi.nlm.nih.gov/compound/11167602)

향후 기술 확인이 승인되면 처리·대조의 물리 명부와 유전자 매핑을 먼저 잠그고, 같은 모델에 같은 정보 계약과 화합물 전체 제외를 적용한다. 개별 연구의 대조군으로 계산한 방향 또는 순위를 같은 지지에서 비교할 수 있다. 서로 다른 발현 척도의 RMSE를 한 표에서 직접 합산할 근거는 확보되지 않았다. 성인 간암에서 기존 VC보다 우수하다는 주 판정은 독립 위치·배양을 갖춘 별도 공통 평가가 필요하다. [96군 파일럿 배치](HCC_PILOT_LAYOUT.md) · [확인 규모 산정](HCC_CONFIRM_DESIGN.md)

새 모델의 학습이나 채점과 예약 자료 접근 없이 마쳤다. 공개 microarray의 일부 상세 metadata 응답에 포함된 값 표는 검사에서 폐기했고, 보존한 자료는 메타데이터와 구조 카탈로그 대조뿐이다.
