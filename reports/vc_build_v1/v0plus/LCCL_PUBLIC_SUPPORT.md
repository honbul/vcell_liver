# Caruso LCCL 추가 공개 검증 후보 감사

저자 연구실의 2019 LCCL 공개 카탈로그에서 세포 목록, 처리 전 FPKM, 약물 카탈로그와 AUC 파일을 내려받아 SHA-256을 고정했다. 원시 WES·RNA-seq 등록은 EGA이고 이번 감사의 근거는 연구진이 별도로 배포한 가공 테이블이다. 파일별 재배포 라이선스는 확인되지 않아 원본은 G1에 보관한다. [원논문](https://doi.org/10.1053/j.gastro.2019.05.001) · [연구진 공개 파일](https://lccl.zucmanlab.com/download.html) · [EGA 등록](https://ega-archive.org/studies/EGAS00001003536)

| 지원 기준 | 모델 | 계통 |
|---|---:|---:|
| 공개 카탈로그 | 34 |  |
| 성인 HCC, FPKM·AUC ID 일치 | 28 | 26 |
| GDSC2·CTRPv2·PRISM 학습에 없는 계통 | 7 | 6 |
| 기존 LIMORE 자료에도 없는 계통 | 3 | 2 |
| 첫 파일 미리보기 노출까지 제외 | 3 | 2 |

계통은 Cellosaurus의 같은 환자 및 파생 계통 그래프를 탐색하고 DepMap의 모델 ID, 환자 ID, 별칭과 대조했다. 완전히 신규인 두 계통은 HCC-1.1·HCC-1.2의 공통 환자 및 HCC-3이다. 중복 유전자 기호를 제외한 소스와의 발현 교집합은 17,281개다. 31개 단일 약물 중 동일 제조사·카탈로그 번호로 기존 학습의 정확한 구조에 연결된 약물은 15개였다. Sorafenib, regorafenib, cabozantinib, lenvatinib이 포함된다. 확인 분석의 계통 5개 기준에 미달하므로 추가 독립 성적은 산출하지 않았다.

초기 헤더 확인에서 CR로 끝나는 파일을 LF 기준으로 읽어 Huh7, SNU739, Hep3B, Huh6, HepG2의 실측 일부가 출력됐다. 해당 세포와 연결 계통을 별도 노출로 기록했다. 그 뒤 감사는 AUC의 헤더와 첫 열의 표본 ID만 읽었다. [해시 및 표본별 감사](lccl_public_support.json)
