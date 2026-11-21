# SNU-423의 동일 약물 외부 RNA 교차 실험 명세

P4에서 이미 평가한 Tahoe SNU-423 5 µM·24 h·최소 20세포 203군을 사용한다. 확장 메타데이터 감사의 적격 100군(34·31·35)에는 GSE92742 비간계열 24 h, 실측 5–20 µM의 동일 canonical parent RNA를 입력으로 사용한다. 실측 5 µM 일치는 2군이므로 농도까지 같은 전이로 부르지 않는다. 허용된 161개 공통 landmark와 물리 웰 명부를 그대로 잠근다. R18 및 다른 봉인·격리·예약 표본은 쓰지 않는다. 같은 약물의 외부 실측과 Tahoe의 비평가 화합물 레이블이 필요한 개발 조건이다. [지원 계약](SNU423_DONOR_SUPPORT_EXPANDED.md)

- 각 source parent×세포주 안에서 실측 용량이 5 µM에 가장 가까운 donor만 사용한다. 동일 RNA plate의 해당 source vehicle 역할 평균을 빼고, 웰·plate·세포주 순으로 균등 평균한다. Tahoe의 A/B 회전에 각각 GSE92742의 잠긴 A/B 역할을 사용한다. 로그 밑 환산은 `ln(2)`다. 원 161 유전자와 판본·연결을 확인한다.
- Tahoe 정답은 원 P4처럼 같은 물리 웰 내 적격 세포주의 DMSO 대비 평균을 뺀 SNU-423 반응이다. 각 outer fold의 적합과 채점에는 서로 반대 물리 DMSO를 쓰며, held parent와 그 웰의 발현은 별도 점수 단계까지 제공하지 않는다. 적합 fold의 5 µM 위치 기준선을 원 `PositionBank`로 계산하고 위치를 공유하는 다른 plate만 사용한다. 적합 행 자신과 동일 parent를 제외한 위치 추정의 배율을 전체 적합 행의 pool 대비 최소제곱으로 [0,2]에서 맞춘다. 이 위치 기준선을 후보와 donor-only 기준선 모두에 준다.
- `donor_scalar`는 공유 위치 대비 잔차에 source 원벡터를 0–2 최소제곱 배율로 적합한다. `donor_rank4`는 supported fit parent source 벡터의 평균·PCA 4축과 고정 ridge 1을 쓴다. source PCA 점수는 미보유군에서 0으로 놓고, 전체 적합 pool 중심을 뺀 뒤 parent 균등 가중 least squares로 공유 위치 잔차를 예측한다. 미보유군에서는 두 arm 모두 공유 위치 기준선으로 돌아간다. `donor_rank4_shuffle`은 fit/held 각쪽의 지원을 유지한 seed `23017+fold`의 다른 parent 순환 할당으로 원벡터만 바꾼다. rank·ridge·용량·세포를 score로 고르지 않는다.
- B0와 원 P4 형제 복사, 기존의 지표별 점추정 최선이던 State와 위치를 받은 State 예측 두 가지를 그대로 같은 행에 둔다. 두 State 예측에 **같은 donor_scalar의 RNA 보정 성분**을 더한 짝을 별도 비교한다. 이 추가 비교의 State 학습은 과거 동결 상태다. 새 RNA 입력으로 State 전체를 재적합한 모델군과의 비교는 제공하지 않는다.
- 기존 P4대로 먼저 전체 plate·용량 pool을 고정해 중심화한다. 각각의 지표에 대한 대비 RMSE와 pooled R²는 전체 203군 fallback, 보유군 100군, 보유군 중 근접 구조 Tanimoto <0.7에 한해 별도 집계한다. 같은 parent 행과 DMSO 역할을 10,000회 가중하고, 보조로 scaffold 가중을 보고한다. 전 집합의 고정 pool 중심을 재표집마다 바꾸지 않으므로 이 구간은 조건부 후속 분석이다. 지원 100군의 rank4−scalar와 rank4−RNA 보정 State 둘의 지표별 최선을 주 비교로 한다. 각 비교에서 ΔRMSE 99% 상한 <0 및 ΔR² 99% 하한 >0이어야 하고, 구조 약지원층과 shuffle·fallback도 따로 공개한다. 기존 판정에는 소급하지 않는다.

경로는 예측 → SHA-256 동결 → 별도 채점이다. 주 후보는 등록한 하나의 저랭크 사상이다. 실제 24 h SNU-423 성능을 본 뒤 이 단계의 설정을 변경하지 않는다.
