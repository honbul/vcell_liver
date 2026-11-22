# HepaRG 고농도에서 동일 약물 비간계열 RNA의 추가 가치

기존 P3와 위치 보정 반복 분석을 보고 등록하는 DEV 후속이다. GSE296254 24 h·실제 ≥10 µM의 기존 parent 3-fold, 풀 대비, T3의 fit-only 2,000 유전자, 동결 위치·처리 기준선과 구조·용량 모델을 유지한다. GSE284321은 기존 자료의 보조 분석이다. 간암 외부 확인은 별도다.

1. 메타데이터 감사에서 GSE92742 24 h·실측 5–20 µM·비간계열 QC 물리 처리 웰을 HepaRG canonical parent와 일대일로 연결한다. 구조 키가 충돌하면 제외한다. R18 예약 처리·물리 별칭·등록 대조·B donor 및 그 대조를 전부 차단한다. source에서 같은 parent·세포주별 10 µM에 가장 가까운 실측 농도 하나를 사용한다. 양쪽 연구의 어느 fold에서도 ≥100군 전체, fold별 ≥20군 지원이 없는 연구의 새 모델은 실행하지 않는다. HepaRG에서 끝까지 같은 support 규칙을 쓴다.
2. 각 비간계열 source plate의 지정된 물리 DMSO 한 몫을 뺀다. 독립 DMSO 3개 이상이 있는 plate만 허용하며 A/B/C로 별도 추정한다. LINCS landmark와 두 TempO-Seq의 827개 고정 일대일 축, 처리 웰→plate→세포주 균등 합의를 쓴다. query parent의 비간계열 RNA는 B 입력이며 HepaRG 처리 정답의 적합에는 해당 outer-held parent를 제외한다.
3. 주 기준선은 동결 T3 `B_context_plus_position + joint_dose_response`다. 추가 `RNA_rank8`은 HepaRG 적합 parent의 두 내부 방향에서 별도 parent로 얻은 위치·처리 기준선의 잔차를, source 발현의 fit-only PCA 8차원과 ridge 1로 예측한다. 적합의 단위는 parent이며 잘못된 parent 대조는 fit과 held 안에서 각각 고정 seed로 순환 치환한다. RNA 없는 웰의 추가값은 0이다. 물리 pool 내 centering과 결측 fallback이 양립하도록 RNA 예측은 관측 parent가 있는 행만으로 각 plate×dose rank의 parent 균등 평균을 빼고, 관측 2군 미만이면 추가값을 0으로 둔다. 입력 군과 순열군의 가용성은 동일하다.
4. 주 비교는 RNA_rank8 대 동일 기준선과 동일 지원의 parent-shuffle 두 비교다. 둘 다 99% paired parent bootstrap에서 상대 RMSE 개선과 잔차 R² 차이의 하한이 0보다 커야 채택한다. 잔차 R² 분모는 공통 기준선의 잔차 에너지다. 10,000회 parent 재표집으로 세 DMSO 회전과 모든 평가 웰을 함께 묶는다. 공통 입력군과 전체 fallback 결과를 함께 보고한다. 반복 관측을 예측 과정에서 사용하지 않으며, 동결 뒤 기존 반복 결과와 같은 결론 범위를 비교한다.

이전 고농도 구조 비교를 보존한다. 메타데이터 승인 명부 → 허용 source 발현 및 fit-only target 발현으로 예측 → 해시 동결 → 별도 score 순서다. 신규 자료 수집과 예약 발현 접근 없이 실행한다.
