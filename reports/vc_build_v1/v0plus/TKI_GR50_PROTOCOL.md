# 6일 성장률 보정 간암 TKI의 전이 검증

Sagmeister 등은 소라페닙·렌바티닙·레고라페닙·카보잔티닙을 6일 처리하고 기저·vehicle 세포수를 사용해 GR50와 최고 농도의 GRmax를 산출했다. 저자 Table 1의 성인 HCC 7모델을 평가한다. HLE·HLF는 같은 공여자이므로 6개 계통이다. 소아 HCC Hep3B와 hepatoblastoma HepG2는 계통 보류 명부에 포함하고 성인 평가에서 제외한다. 발현은 저자와 같은 LIMORE 공개 기저 RNA다. 기존 AUC 스크린에 다수 계통이 있으므로 DepMap 환자키, Cellosaurus 같은 환자·파생 관계 및 동일 모델에서 연결한 모든 계통을 전 약물·전 소스 학습에서 보류한다. 논문 검색에서 일부 정성적 약물 관찰이 보였음을 기록한다. Table 2의 정량 표는 등록·적합·동결 후 별도 채점에만 사용한다. [저자 공개 논문](https://epub.ub.uni-muenchen.de/106583/) · [LIMORE GSE97098](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE97098)

## 모델

- 정정 GDSC2·CTRPv2·PRISM의 AUC를 사용한다. 기존 고정 설정의 약물 평균, 약물별 Ridge, CellHit, CODE-AE inductive, Velodrome inductive, V0+ inductive 및 구조 커널을 같은 등록 행에 적합·예측한다. 구조 커널은 RNA·PROGENy/Hallmark/DoRothEA·ECFP·표적·Chronos 및 변이/CNV의 RNA 기반 보조 표현을 기존과 같이 사용한다.
- `growth_adjusted`는 구조 커널의 학습 잔차에서 선형 증식률 성분을 제거한다. Sanger의 day 4 / day 1 대조군 신호 비율이 1보다 크고 반복이 3회 이상인 학습 세포를 사용한다. 동일 세포·접종밀도의 비율은 로그2 변환 후 세포별 중앙값으로 통합한다. 적합 세포의 평균·표준편차를 사용해 조정한다. 적합 스크린×약물에서 측정 세포가 30개 이상이면 그 집합의 증식률을 중심화하고 잔차에 대한 벌점 50의 단변량 Ridge 기울기를 계산해 제거한다. 미측정 행에는 0을 적용한다. 원래 약물 앵커, 벌점 4와 배율 1을 유지한다. `growth_shuffled`는 같은 조직 안에서 증식률만 순열한 대조군이다. seed는 43675다. 모든 모델에 새 환자 증식률 입력은 들어가지 않는다. Sanger 대조군 측정은 저자 실험의 직접 증식률이 아니다. [공개 대조군 증식률 파일과 규약](https://depmap.sanger.ac.uk/documentation/cell-models/growth-rate/)

## 채점과 판정

- 숫자 GR50은 Fig. 1의 시험 범위로 제한하고 `−log10`으로 변환한다. 값에 `>`가 붙거나 상한을 넘으면 그 상한에 둔 뒤 우검열 수를 보고한다. 상·하한은 sorafenib 0.31–5, lenvatinib 0.25–20, regorafenib 0.31–5, cabozantinib 1.25–20 µM다. GRmax는 부호를 뒤집고 약물 내 기술 통계만 보고한다. 서로 다른 최고 농도에서 측정된 GRmax의 약물 간 순위를 해석하지 않는다.
- 공여자별 동일 가중 평균을 먼저 계산한다. GR50의 약물 내 Spearman·Pearson과 공여자 내 Spearman·Top-1 일치를 보고한다. 네 약물에서는 NDCG@5와 Precision@5를 적용하지 않는다. 기존 LIMORE AA를 같은 행에서 한 번 대조해 두 실측의 약물 내 방향 일치도와 모델의 양쪽 성적을 기술 보고한다. 투약 시간·농도·읽기 방식 차이가 함께 존재한다.
- `growth_adjusted`와 `structured`, `growth_shuffled`, `drug_ridge`의 짝지은 차이를 비교한다. 보조 증식률 가설은 GR50 약물 내 Spearman·Pearson의 공여자·약물 교차 재표집 95% 하한이 세 대조에서 모두 0을 넘고, 공여자 내 Spearman 하한이 모든 대조에서 −.005보다 높을 때 지지한다. 6개 계통의 기술 검증으로 기록하며 LIMORE 50계통·LICOB 38공여자의 외부 판정과 배포 규칙을 유지한다.

정량 반응은 G1에 보관한다. 저자 PDF SHA-256은 `4aa52d413a0a7732caccfcce6b837d167c6b83f5427c3562c16d6d5ca297842d`이고 Sanger 원본은 `cdbc9f3c6c7b2388ade1780bc93feb6579f6868b73ba3dbc80dfb57d60c0a4dc`이다. [Sanger 내부 연구 사용 조건](https://depmap.sanger.ac.uk/documentation/data-usage-policy/)
