# 같은 6시간 실측을 받는 HEPG2 시간 모델과 State 비교

HEPG2 GSE92742의 기존 scaffold 차단 T4에서 단순 배율 대비 개선이 유지됐다. 같은 약물·실측 농도의 6시간 반응을 입력으로 받을 때, 고정 저랭크 연산자와 공개 State Transition 적응의 24시간 예측을 비교한다. 고정된 개발 평가 화합물 132개·물리 웰 268개, 978개 landmark, 자체 RNA plate 비중복, 세 vehicle 역할 및 기존 outer fold를 유지한다.

- 각 fold×plate×role에서 기존 T4 scaffold 차단과 exact-dose fit 쌍을 동일하게 재현한다. State에는 같은 조건의 fit-role 6 h 대비 벡터를 1개 토큰으로 주고 전체 샘플에 같은 6→24시간 개입 코드 1을 준다. L1000의 단일 웰 평균 측정에 맞춘 적응이다. 6시간 실측의 가용성, 적합 24 h 웰의 가용성 및 query 정답 차단은 기존 저랭크 모델과 동일하다. 별도 구조 특징은 두 모델 모두 예측 입력에 쓰지 않는다. 구조는 scaffold 분할에만 사용한다.
- T2 State의 고정 공개 구현 및 버전, GPT2 backbone·hidden 256·잔차 출력·AdamW(학습률 0.0003, weight decay 0.001)·epoch 40·seed 42·배치 최대 128을 사용한다. 기존 State의 단일세포형 DMSO 토큰과 다른 1토큰 적응이며, 학습 정답의 24 h 벡터와 같은 좌표에서 parent 균등 MSE로 적합한다. 목표 세포·농도와 시간은 해당 T4 쌍에서 주어진다. 24 h query, score DMSO, 체크포인트나 출력 배율 선택은 학습에 제공하지 않는다.
- 주 비교는 기존 동결 `purged_operator` − `State_6h`다. parent-macro RMSE의 차이 99% 상한 <0 및 pooled R² 차이 99% 하한 >0을 요구하고, 실용 기준인 상대 RMSE 0.5% 및 R² +0.5%p도 함께 적용한다. 앞선 동결 단순 배율과 6시간 직접 복사, 독립 24시간의 비보정 복사는 기술 참조다. 10,000회 짝 parent 구간 및 같은 scaffold 군집 민감도를 모두 기록한다.

명세 → State 예측과 기존 입력·비교군 검증 → 해시 동결 → 별도 채점 순서다. GPU 동시 사용은 최대 두 개다. R18 예약, 신규 자료 및 성인 HCC 표본은 쓰지 않는다. 이전 공개 모델의 우위에 대한 판정은 이 정보 계약과 이 State 적응 범위에 해당한다.
