# 첨부 리뷰의 HCC 섭동 사례 접근성

리뷰가 인용한 [Zhou et al., Molecular Cancer 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8722008/) 원문을 확인했다(DOI10.1186/s12943-021-01473-w). Huh7 parental과 장기 sorafenib 노출 후 resistant 집단의10X scRNA-seq이며 필터 후4776cells를 분석했다. 논문의 data availability는 corresponding author에게 reasonable request로 제공한다고 명시한다. 확인한 원문에는 즉시 내려받을 수 있는 raw-count accession이 제시되지 않았다. 저자에게 연락하지 않았고 접근 가능한 accession을 추정해 만들지 않았다.

이 endpoint는 장기 내성 선택의 집단 차이다. 이를 acute drug response나 같은 세포의 전후 관측으로 간주하지 않는다. 현재 실제 실행 가능한 경로는 HepG2 CRISPRi와 mouse liver TCDD이며, 두 데이터 모두 사람 성인 HCC 약효 검증과 구분한다. 본 사례는 `blocked_author_request_only`로 남긴다. 새 접근 자료가 생기면 별도 protocol과 실측 endpoint를 먼저 고정해야 한다.
