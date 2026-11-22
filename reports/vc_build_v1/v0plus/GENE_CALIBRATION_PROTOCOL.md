# 유전자별 RNA 분포 보정의 LIMORE·LICOB 전이

같은 유전자의 상대 발현이 서로 다른 측정 플랫폼에서 보존되는지 검증한다. 학습 자료는 정정한 GDSC2·CTRPv2·PRISM20Q2 단일 약물 65,678행이다. 이전과 같은 LIMORE 성인 HCC 50계통·68약물, LICOB 오가노이드 43개·성인 공여자 38명·53약물을 별도로 평가한다. 이미 채점한 두 코호트의 사후 기술 비교다. [LIMORE 계통 감사](../data_function/VCB01_PUBLIC_CROSS_SUPPORT.md) · [LICOB 넓은 RNA 축](LICOB_WIDE_RNA_RESULTS.md)

- RNA는 각 코호트와 학습원의 공통 유전자에서 샘플 내 평균 순위로 변환한다. 같은 계통·공여자의 여러 RNA는 유전자별 참조 분포에 한 평균값으로 기여한다. 커널에는 각각의 샘플 RNA를 사용한다. LICOB 정답과 예측은 채점에서 공여자별로 평균한다.
- 기존 구조 커널, 약물 커널, 약물 평균 anchor, 관측 가중과 벌점 4를 유지한다. 기존 세포 커널 .60과 아래 변환에 대한 추가 세포 커널 .40을 결합한다. 추가 커널은 소스 유전자 중심·표준편차 하한 .05, 표준화 후 절댓값 상한 8, 소스 평균 자기 유사도 정규화, 선형과 RBF 폭 2의 50:50 결합이다. 경로, TF, Chronos 및 유전체 보조 표현은 기존 커널에 유지한다.
- augmented_raw는 원래 샘플 내 순위를 쓰는 추가 커널이다. cohort_centered는 같은 학습 커널과 계수에 평가 RNA의 공여자 균형 유전자별 평균 차이만 보정한다. source_ecdf는 동일 유전자 순위에 소스 계통 참조의 경험 CDF를 적용하며 평가 샘플 하나씩 추론할 수 있다. CDF는 미만 관측과 이하 관측 비율의 평균이다. cohort_ecdf는 동일하게 적합한 계수에 각 평가 코호트의 공여자 균형 참조 CDF를 쓰는 RNA 재정렬이다. 두 cohort 조건은 transductive로 표기한다. 약물 정답과 평가 약물 평균은 RNA 보정에 사용하지 않는다.
- 주 후보 cohort_ecdf는 두 코호트 모두에서 source_ecdf, cohort_centered, 기존 구조 커널 structured 대비 약물 내 Spearman·Pearson 차이의 공여자×약물 짝지은 95% 구간 하한이 모두 0을 넘고, 계통 내 Spearman·NDCG@5의 짝지은 하한이 모두 −.005보다 높으면 승격한다. 약물 평균, 약물별 Ridge, CellHit·CODE-AE·Velodrome·V0+를 동일 행에 놓고 Precision@5와 주요 약물군도 보고한다. source_ecdf를 augmented_raw 및 structured와 비교해 단일 RNA에 적용 가능한 버전의 기술 결과를 기록한다.
- 입력과 기존 예측 해시를 검증하고 코호트별 새 비교를 GPU에서 생성해 동결한다. 별도 작업에서 정답을 처음 연결해 채점한다. 원본, 모델과 환자별 예측은 G1에 둔다.
