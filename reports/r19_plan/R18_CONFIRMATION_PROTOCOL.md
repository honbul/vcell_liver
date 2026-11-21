# R18 확인 입력 사전 명세

이 명세는 R19에서 감사한 R18 평가 행과 R17 선택 규약을 연결한다. 원 후보·격자·손실·anchor·비교군·판정을 유지한다. 입력 명부는 처리 및 대조군 발현값을 읽기 전에 고정했다. [R17 명세](../r17_lincs/PROTOCOL.md) · [지원 감사](progress_reconciliation.md) · [입력 잠금 영수증](data/preflight/r18/receipt.json)

## 평가 행과 적합 규칙

| 대상 | 내부 적합·선택 자료 | 고정 평가 자료 |
|---|---|---|
| C1 주 확인 | C1의 해당 scenario 적격 HEPG2 parent 중 outer-fit군 | CMap 2020 HEPG2 24 h 5–20 µM의 R16 동일 QC 행. R16 이전 규약 채점과 B1·R8 동일 웰 노출을 표시한다. |
| C2 용량별 보조 | GSE70138의 검증된 RNA 좌표와 CMap 적격 행을 하나의 HEPG2 pool에서 적합. 공통 parent split·R17 특징을 적용한다. | 같은 pool의 held parent를 원 µM별로 보고한다. 100 parent 미만 층은 기술 분석이다. 농도 bin을 새로 만들지 않는다. |
| C3 주 기술 6 h 및 24 h | 각 시간의 GSE92742 HEPG2 서명상 3 µM, HUH7 실측 용량 범위 2.2972–3.825 µM, QC와 물리 DMSO 3역할 적격 웰. 적합과 내부 선택에 HEPG2 처리만 사용한다. | 같은 시간의 HUH7 서명상 3 µM. |
| C3 HUH7 추가 기술 | 같은 HEPG2 적격 pool에 HUH7 outer-fit parent를 추가한다. 원 R17 방식대로 같은 parent 내분할을 두 세포주에 적용해 두 세포주의 inner-select 행으로 선택한다. | 주 기술과 같은 HUH7 평가 웰. |

C3 HUH7 추가 설계는 두 세포주에 동일한 CRC32 parent 규칙을 쓴다. HUH7-only 내부 선택으로 제한하면 B 24 h fold 2에 선택군이 없었다. 결합 풀 전체에 원 R17 내부 분할 규칙을 적용하면 이 fold에는 HEPG2 3군·14웰의 내부 선택 지원이 있다. 모든 fold에 같은 규칙을 적용한다. HUH7의 미처리 DMSO는 예측에 필요한 fit 맥락과 분리된 채점 정답을 만드는 데 사용한다.

C1과 C2의 위치 추정은 연구 release별 동일 좌표 및 다른 물리 plate를 사용한다. C3는 R16 확인 코드의 세포주별 위치 추정을 유지한다. HEPG2-only의 HUH7 위치 예측은 0이다. 각 측정은 물리 RNA plate와 well 기준으로 한 번만 등록한다. C2의 GSE70138와 CMap 양쪽 행에는 공동 outer parent 분할을 적용한다.

시나리오 A는 적격 구조 및 고정 R17 특징을 사용한다. B는 A에 허용된 비간계열 동일 parent 실측을 추가한다. C1과 C2의 B donor 지원은 평가 행의 등록 release에 대응하며 각각 5–20 µM 및 0.04–3.33 µM, 24 h이다. C3는 GSE92742에서 같은 시간과 HUH7의 실측 농도 범위에 있는 등록 donor를 쓴다. B의 donor는 모델 입력 전용이다. 평가 parent의 hepatic 처리 라벨은 해당 fold의 적합과 선택에서 모두 배제한다.

## 역할, 동결, 판정

원 RNA plate의 CRC32 물리 역할을 `ABC`, `BCA`, `CAB`로 순환한다. 안쪽 fitting과 최종 재적합의 delta와 특징은 fit DMSO를 쓰고, 내부 선택 정답은 select DMSO를 쓴다. 예측 프로그램에는 score treated 정답과 score DMSO가 없는 입력 묶음을 제공한다. 각 예측의 물리 웰·유전자축·선택 영수증을 동결하고 별도 채점이 해당 해시를 검증한다. 여러 역할 회전을 parent 단위로 함께 재표집한다.

주 후보는 커널+FiLM, 보조는 커널+concat과 R15 커널 단독이다. 모든 모델은 해당 fold에서 새로 적합한다. 후보 및 비교군 선택은 원 R17 혼합 손실과 gain grid `0, .125, .25, .5, .75, 1`을 사용한다. R17의 레거시 순차 선택, 커널 선택, 결합 계수 곱집합, 두 anchor 정의를 유지한다.

C1에서 B0, B_pos, 배율을 적용한 전체·출판군·프로젝트별 최선 VC, R15, B의 B_cell×g를 비교한다. 10,000회 parent 짝 재표집의 ΔRMSE 99% 상한 < 0 및 ΔR² 99% 하한 > 0을 모든 등록 비교에 적용한다. A와 B 모두 통과하면 ‘확인됨’, B만 통과하면 ‘전이 조건에서만 확인됨’, 그 외에는 ‘확인되지 않음’을 쓴다. 커널 단독도 해당 비교군과 함께 제시한다. C3 및 소수 parent C2 층은 기술 구간을 보고한다.

동결 뒤에 코호트별 교차 plate 반복 기준 및 도달률, 상위 5·10·20 parent 오차 기여와 제외 민감도, 선택 배율, score vehicle 방향 cosine, R15–R18 동일 비교 이력을 추가한다. 위치 donor 지원과 정량 B_pos 대비 B0는 구별한다. 정량 비교는 동결 후 채점에서 보고한다.

## 메타데이터 실행 확인

CPU 작업 `30670`이 R19 감사 파일의 해시, 7,295개 고유 대상 물리 웰, 2,311개 대조군 물리 웰의 R17 CRC32 배정, B donor 차단, C3 HEPG2의 예약 측정 분리 및 outer·inner parent 분리를 검증했다. 모델·시나리오별 focus를 합친 12개 케이스, 3개 fold, 3개 역할 차례로 108개 조합의 지원이 있다. [fold별 명부 요약](data/preflight/r18/support.tsv) · [G1 원장](data/preflight/g1_runs.tsv)

CPU 배열 `30679`는 CMap 2020, GSE70138, GSE92742에서 평가·적합·내부 선택의 고유 처리 웰 8,083개와 물리 DMSO 2,311개를 원 Level 3의 ID에 연결했다. 별도 B 입력은 donor 처리 19,454웰, donor DMSO 19,749웰이었고 957개 donor plate 전부에서 두 대조군 몫에 지원이 있었다. R15/R16과 같은 순서로 donor DMSO를 CRC32 정렬의 두 몫에 나눴다. 이 몫은 모델의 fit/select/score 역할과 분리된 비간계열 입력 구성용이다. 고정 978개 landmark는 세 자료의 원 ID에 모두 대응했다. 원 매트릭스에서 well·gene ID 메타데이터 두 경로만 열었고, G1에서 발현 배열이 아예 없는 테스트 파일로 동일 코드의 동작을 확인했다. [Level 3 지원표](data/preflight/r18/index/support.tsv) · [G1 작업 원장](data/preflight/g1_runs.tsv)
