# HEPG2 시간 연산자의 scaffold 차단 재적합

기존 T4의 6→24 h 이득이 적합에 평가 화합물의 가까운 화학 계열이 남았는지에 의존하는지 검정한다. 기존에 채점한 GSE92742 HEPG2 132개 canonical parent·268개 24 h RNA 웰에 대한 개발 후속이다. 같은 6 h 실측값과 978개 landmark, 실제 농도, 물리적 채점 plate 제거, 승인된 T4 명부, 세 DMSO 역할 및 세 parent fold를 그대로 쓴다. 표현형·외부 RNA 뷰와 신규 자료는 추가하지 않는다.

- 기존 T4 canonical parent SMILES에서 원자와 결합을 보존하는 Bemis–Murcko scaffold를 생성한다. scaffold를 만들 수 없는 비고리 parent는 각각 별도 군으로 둔다. 해당 outer fold 평가 parent 중 하나와 scaffold가 같은 모든 source parent를, 모든 시간·농도의 연산자 적합에서 차단한다. 평가 plate를 제외한 뒤 남은 원 적합 집합에서 제거한다. 같은 수 제거 대조는 `CRC32("T4scaffold|fold|plate")`를 seed로 해당 plate의 전체 적합 parent에서 균등하게 한 번 뽑는다. 두 fit과 원 적합 부모 수 및 제거 수를 기록하고, 평가 parent와 scaffold 교집합을 검사한다.
- `purged_operator`와 `purged_scalar`는 동일한 scaffold 차단 source 데이터를 사용한다. `count_operator`와 `count_scalar`는 같은 수 제거 source 데이터를 사용한다. `full_operator`와 `full_scalar`는 그 plate의 전체 적격 source를 사용한다. 연산자는 기존 T4에 쓰였던 rank 16, ridge 1로 모두 고정한다. 조건별 웰 평균과 parent 균등 가중 및 scalar 0–2 제한은 T4와 같다. rank를 지원하지 못하거나 fit이 17 parent 미만인 경우 등록 범위에서 미실행을 기록한다. 기존 동결 `identity6`을 모든 arm의 동일 입력으로 쓰며 `direct24`는 기술 참조다.
- 주 비교는 `purged_operator` 대 `purged_scalar` 및 `identity6`다. 두 비교 각각에서 parent-macro RMSE 차이의 99% 상한 < 0, pooled R² 차이의 99% 하한 > 0이 필요하다. 0.5% RMSE 개선과 +0.5%p R² 실용 문턱을 구간 및 점추정과 함께 낸다. `purged_operator` 대 `count_operator` 및 `full_operator`와 기존 24 h 반복 참조는 기술 비교다. 10,000회 짝 parent 재표집을 원 T4 결과와 동일하게 사용한다. 추가 scaffold 성분 군집 재표집은 기술 민감도다. 동일 scaffold가 없는 경우에도 ECFP4가 유사할 수 있으므로 이를 모든 표현 공간의 저유사도 검증으로 확대하지 않는다.

명세 → fold·역할별 예측 → 동결 → 기존 T4 채점 명부를 사용한 독립 채점 순서다. 예약 발현과 새 다운로드는 사용하지 않는다. 새 24 h 지표로 rank나 ridge를 고르지 않는다.
