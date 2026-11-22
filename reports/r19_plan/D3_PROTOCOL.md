# D3. 적합 DMSO만 사용하는 보정

D1의 동일 표본수 A 개발 지원을 사용한다. GSE92742 HEPG2 6 h 처리 6,773개 물리 웰과 1,862 parent, 물리 DMSO는 판마다 역할별 같은 수로 총 486개다. D3a·D3b·D3c는 같은 위치 anchor와 구조 커널, 적합·선택·평가의 세 독립 역할, parent 분할을 쓴다. 입력은 D1이 이미 분리한 predictor-only 묶음과 물리 역할 ID다. Score DMSO는 동결 뒷단에만 들어간다.

- D3a는 균등 fit DMSO의 원 평균을 쓰는 full-gene MSE FiLM이다.
- D3b는 fit DMSO를 물리 ID CRC32 순으로 번갈아 두 집합에 놓는다. 같은 처리 웰에서 각 view의 DMSO 평균을 뺀다. 위치 anchor와 기존 fit-only pool offset을 두 view에 공유하고, 같은 fit-control PCA로 두 입력을 변환한다. 관측 대조군 수 비율로 가중을 배분해 처리 웰당 전체 가중과 평균 잔차를 보존한다. 기존 backbone, 잔차 scale, epoch당 update 수를 유지한다.
- D3c는 plate별로 중심화한 fit DMSO의 공분산에서, plate당 같은 가중의 rank `1, 2` 방향을 산출한다. 약물 예측의 공통 위치 성분·구조 커널·fit-only pool offset을 보존하고, FiLM 잔차만 `0, .25, .5`의 비율로 줄인다. Rank·강도는 선택 역할에서만 고른다.

모든 arm에서 R17 DEV의 고정 checkpoint, 학습률과 두 결합 gain을 독립 select 정답의 공동 손실로 고른다. 각 arm의 `parent_shuffle`은 FiLM의 화학 입력을 fit과 query에서 서로 독립적으로 재배열하고 재학습한다. 위치와 구조 커널은 공유하므로 이 대조는 신경망 경로의 화학 입력 기여를 측정한다.

공동 주 지표는 D3b·D3c 각각의 D3a 대비 parent-macro RMSE와 pooled R², 10,000회 짝 parent bootstrap 양측 99%다. 동일 실측 용량 pool에 대해 대비 RMSE를 기술하며, landmark 전체의 효과 부호 일치와 `Full−Shuffle` 차이를 기록한다. 부호 일치는 기전의 인과 검정과 별도다. 기존 최소 유용 개선인 상대 RMSE 0.5%, R² 0.5%p를 적용한다. 9개 조각의 예측을 한 번에 동결하고 독립 작업에서 균등 score DMSO로 채점한다.
