# 독립 10 µM·8원천 HCC 파일럿 배치와 실측 전 준비

**서로 다른 scaffold의 화합물 96군에 16개 plate의 배치를 확보했다.** CMap에서 같은 약물·실측 10 µM·24 h 비간계열 원천을 약물당 8개 세포주에서 고정했다. 각 약물의 SNU-423 및 HUH7 처리 좌표 6개를 해당 원천의 모든 RNA 웰 인덱스와 분리했다. 배치는 실제 검정에 앞서 독립 배양 변동과 반복 상한을 추정하는 파일럿이다. [배치 명세](HCC_OFF_POSITION_PILOT_PROTOCOL.md) · [화합물 자리표](data/hcc_off_position_pilot/audit/slots.tsv) · [Plate 배치](data/hcc_off_position_pilot/audit/plate_layout.tsv)

| 기존 성인 HCC와 R18 노출 제외 후 | 비간계열 원천 하한 | 위치 지원 parent | 서로 다른 scaffold |
|---|---:|---:|---:|
| Launched·Phase 2·Phase 3 | 8개 | 315 | 272 |
| 모든 등록 단계 | 8개 | 441 | 373 |
| Launched·Phase 2·Phase 3 | 16개 | 23 | 22 |
| 모든 등록 단계 | 16개 | 29 | 28 |

8개 계약의 96군은 Launched 54군, Phase 2 28군, Phase 3 14군이다. 약물당 선택한 원천은 서로 다른 RNA plate 9–54개에 있었다. 이전 16원천 감사 뒤에 등록한 저용량 JHH 표적 parent를 새로 제외하니 동일 단계의 16원천 scaffold가 37군에서 22군으로 줄었다. 등록 단계를 넓힌 16원천 지원도 96군에 미달한다. 기존 16원천 계획은 그대로 보존했고 이 파일럿의 8개 원천 조건은 별도다. [지원 원표](data/hcc_off_position_pilot/audit/support.tsv) · [이전 16원천 감사](HCC_10UM_DONOR_PILOT_RESULTS.md)

각 세포주의 8개 배양 블록 자리에서 약물당 서로 다른 블록에 3회, 가장자리 1회와 내부 2회로 배치했다. 두 세포주 합계 처리 576웰과 물리 DMSO 192웰이다. Plate당 처리는 36웰이며 세 vehicle 역할에는 각각 사분면당 한 웰이 배정됐다. 원천 인덱스와 처리 좌표의 겹침은 0건이다. 비교하는 위치는 96웰 목표와 기존 CMap 원천의 행·열 라벨이다. 실제 배양 독립성과 실험 플랫폼에서의 물리 배치 적합성은 측정 전에 확인해야 한다. [Plate 검산](data/hcc_off_position_pilot/audit/plate_checks.tsv) · [원천과 겹치는 금지 위치](data/hcc_off_position_pilot/audit/forbidden_positions.tsv)

배치 감사 `31730`은 메타데이터만 사용했다. Seed `191197`과 원표 SHA-256을 검증했다. 새로운 표적 발현 자료를 수집하거나 물질을 주문한 기록은 없다. [감사 영수증](data/hcc_off_position_pilot/audit/receipt.json) · [작업 원장](data/hcc_off_position_pilot/audit/jobs.tsv)

## 실측 전 예측 동결

신규 96군과 scaffold가 겹치는 개발 자료를 제외한 JHH-5 277군·281웰·6개 RNA plate에서 적합했다. 신규 원천 실측은 1,930웰·701개 RNA plate다. 새 표적 발현은 사용하지 않았다. 같은 입력의 8토큰 State, 원평균 단일 배율, 배율 보정 편차 축소 및 대조를 포함한 8개 arm의 DMSO 역할별 `96 × 978` 배열을 동결했다. 동일한 화합물별 예측이 두 새 세포주에 적용된다. [예측 동결 명세](HCC_OFF_POSITION_PILOT_FREEZE_PROTOCOL.md) · [적합과 원천 지원](data/hcc_off_position_pilot/forecast/access/support.tsv)

다른 약물 대조도 96군의 일대일 순열을 고정해, 대체 원천 좌표와 해당 목표의 처리 좌표가 겹치지 않게 했다. 적합과 신규 평가의 정확한 parent 및 scaffold 중복은 모두 0이다. 예측 배열과 접근 명부의 SHA-256, 역할·슬롯·978유전자 순서 및 B0 배열을 검산했다. [대조 순열](data/hcc_off_position_pilot/forecast/access/wrong.tsv) · [독립 접근 검산](data/hcc_off_position_pilot/forecast/access/verified.json) · [동결 영수증](data/hcc_off_position_pilot/forecast/freeze.json)

첫 접근 작업은 앞선 scaffold 차단 분할의 적합 명부를 합치면 305군 전부가 복원된다는 잘못된 가정으로 중단됐다. 기존에 채점한 개발 명부의 고정 전체 행에서 305군을 재구성해 교정했고 첫 시도의 모델 작업은 실행되지 않았다. 수정 후 접근 `31734` → 예측 `31735` → 동결 `31736`은 종료 코드 0이었다. 동결 SHA-256은 `e11da7d92124bf3a3e030c7a334cf79d15625cb050f48f3c23b5ff5ad7a24671`이다. 실제 물질·독립 배양·plate와 978개 표적 측정이 확보되면 별도 채점이 가능하다. 현재 정확도와 외부 확인 판정은 미측정이다. [재실행 원장](data/hcc_off_position_pilot/forecast/jobs.tsv) · [첫 접근 기록](data/hcc_off_position_pilot/forecast/first_attempt.tsv)
