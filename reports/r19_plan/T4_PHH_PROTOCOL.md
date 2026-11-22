# T4 시간 전이의 일차 간세포 기술 검증 명세

GSE92742의 PHH에서 기존 T4 지원 감사가 등록한 67개 parent·104개 실제 농도 조건을 사용한다. PHH는 일차 간세포다. 이 비교는 시간 전이 연산자를 별도 간 맥락 안에서 검증하며 성인 HCC 확인은 R18과 분리한다. 공개 원본을 새로 수집하지 않는다. 등록 R18 처리, 별칭, 채점, B donor 및 donor vehicle을 모두 기존 T4 감사와 같은 방법으로 차단한다.

- T4의 정확한 parent·실제 µM·6/24 h, 양쪽 2개 이상 물리 RNA plate, 시간 사이 plate 비중복, CRC32로 고른 24 h 채점 plate, outer parent 세 분할과 전 plate 배제, 세 vehicle 역할 회전, 978 landmark와 웰 단위 지표를 그대로 쓴다. QC·plate·fold 명부는 발현 접근 전 검증한다. 반복 참조의 donor는 평가 plate에서 완전히 분리한다.
- PHH에는 내부 설정 탐색을 추가하지 않는다. 저랭크 연산자는 기존 T4 후보 가운데 소표본에서도 적합 가능한 rank 16, HEPG2의 84개 조각에서 공통으로 선택됐던 ridge 1로 미리 고정한다. PHH의 해당 24 h 평가 plate와 parent를 뺀 적합군만으로 추정하며, 전체 적합군에서 17 parent 이상과 16 수치 rank를 요구한다. 이 설정으로 67군 모두 적격하지 않으면 하위집단 선택 없이 미실행을 기록한다. 다른 성분은 기존 T4 그대로다. `B0`, `identity6`, `scalar6`, `operator6`, `direct24`, `logdose6`를 같은 행에서 비교한다. 새 rank·ridge 선택과 HEPG2 계수의 PHH 직접 이식은 없다.
- 예측에는 score 정답과 score DMSO 인덱스가 없다. 순서는 메타데이터 지원·명부 고정 → 슬럼 예측 → 해시 동결 → 별도 채점이다. 10,000회 짝 parent 재표집의 99% 구간으로 parent-macro RMSE와 pooled R²를 점검한다. `operator6`가 `identity6`, `scalar6`를 각각 두 지표의 99% 구간에서 앞서면 PHH에서의 시간 보정 개선으로 기록한다. 독립 24 h 직접 반복도 같은 두 지표로 비교하되 별도로 해석한다. RMSE 이득 0.5%, R² 0.5%p 미만의 구간 상한은 실용 규모의 해석에 함께 쓴다. 다른 세포주/실험의 공유 parent 여부와 별개로 PHH 결과는 동일 L1000 자료 안의 별도 맥락 DEV로 둔다.
