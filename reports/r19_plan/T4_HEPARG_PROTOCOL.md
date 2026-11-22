# LINCS 시간 전이와 HepaRG의 동일 조건 기술 비교

기존 GSE92742 HEPG2 T4의 동결 예측을 GSE284321과 GSE296254의 24 h TempO-Seq에서 살펴본다. 두 HepaRG 연구에는 R14·T3·P3 노출이 있다. 메타데이터 감사에서 정확한 canonical parent와 실측 농도, 독립 LINCS 시간 plate 및 HepaRG plate별 3개 QC DMSO를 동시에 요구했다. GSE284321 5군·21웰, GSE296254 3군·9웰이 남았다. GSE296254의 세 화합물은 하나의 parent fold에 속한다.

- 원 T4의 독립 채점 24 h RNA plate와 그 plate를 뺀 24 h donor를 그대로 유지한다. 동일 농도 6 h donor, T4 저랭크 6→24 h 예측, 직접 24 h donor의 세 동결 배열만 사용한다. 세 DMSO 역할을 그대로 유지한다.
- P3의 기존 Entrez–Ensembl 일대일 대응에서 두 TempO-Seq assay에 공통인 827개 랜드마크를 순서까지 고정한다. 같은 패널에서 세 공유 예측을 독립 LINCS 24 h 웰과 각 HepaRG 24 h 웰에 적용한다. GSE284321과 GSE296254를 별도로 낸다.
- HepaRG의 물리 DMSO는 plate별 CRC32(`library`)와 이름 순으로 정렬하고 ABC를 순환 배정한다. 독립 유전자 좌표에서 R14의 전체 vehicle 대비를 원 log1p CP10k로 되돌린 뒤 해당 T4 역할의 score 배정 DMSO 평균을 뺀다. 예측에 HepaRG 처리값이나 DMSO를 사용하지 않는다.
- 유전자의 평균 중심화를 추가하지 않은 cosine을 사용한다. 모델의 배율이나 플랫폼 변환을 고르지 않는다. 물리 웰·역할을 먼저 parent 안에서 평균하고 각 연구의 동일 parent에 대한 세 예측의 평균 및 범위를 제시한다. 3–5개 화합물의 기술 진단이며 p값·확증 판정의 대상은 아니다. L1000과 TempO-Seq의 단위가 달라 두 assay 사이에 원 RMSE를 빼지 않는다. 세포 맥락과 측정법이 함께 바뀐다.
- 메타데이터와 원 T4 동결 배열에서 새 비교 예측을 독립 동결한다. 별도 채점 프로세스에서만 두 assay의 지정 웰·score DMSO 발현을 읽는다. R18 예약값은 접근 범위 밖이다.
