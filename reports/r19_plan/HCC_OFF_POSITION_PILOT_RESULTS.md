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

## 별도 채점 경로 검증

실험 인수 `31739`와 별도 채점 `31740`에서 원 동결 예측을 고정하고 표적 발현만 합성한 입력을 통과시켰다. 인수 단계는 발현을 열지 않았다. 채점은 순서를 뒤집은 두 축을 ID로 맞췄으며, 역할당 DMSO 3웰 조건을 적용해 539개 합성 처리 웰을 평가했다. SNU-423은 288웰·8개 합성 배양, HUH7은 251웰·7개 합성 배양이다. 합성 처리 웰 수는 실험 표본 수가 아니다. 순서와 인수 해시 검증 뒤 parent 및 배양을 함께 고려한 10,000회 재표집을 실행했다. [채점 명세](HCC_OFF_POSITION_PILOT_SCORE_PROTOCOL.md) · [합성 지원](data/hcc_off_position_pilot/score_smoke/score/support.tsv)

합성 입력의 실측 모드 사용, 계획과 다른 화합물, 재사용 배양 시작 ID, 잘못된 인수 잠금 및 해시가 다른 발현 파일을 각각 거부했다. 이 검산으로 배치·측정 내보내기·동결 모델을 잇는 소프트웨어 경로를 확인했다. 실제 독립 배양, assay 호환성과 정확도는 실측 자료가 필요하다. [거부 검산](data/hcc_off_position_pilot/score_smoke/verification.json) · [합성 채점 영수증](data/hcc_off_position_pilot/score_smoke/score/receipt.json)

## 공개 VC 추가 동결

기존 State에 더해 XPert, PRnet, chemCPA, TranSiGen의 96군 예측을 별도로 동결했다. 네 적응 모두 기존 후보와 같은 8개 비간계열 원천을 받았다. 각 DMSO 역할에서 JHH-5 내부 적합 224군과 선택 53군을 쓴 뒤, 전체 277군에서 다시 적합했다. 재적합한 단일 배율 기준선은 원 동결 배열과 정확히 일치했다. 준비 `31758`, 예측 `31759`, 동결 `31760`의 추가 영수증 SHA-256은 `9ba8564065d6854f12e8a12d0bc7c3058303d08280cd634c5b4cb0716d93cfda`다. [추가 비교 명세](HCC_OFF_POSITION_PILOT_NATIVE_PROTOCOL.md) · [추가 동결 영수증](data/hcc_off_position_pilot/native/forecast/freeze.json) · [실행 원장](data/hcc_off_position_pilot/native/forecast/jobs.tsv)

추가 채점 `31774`는 원 동결과 추가 동결을 같은 합성 인수 명부에 연결했다. 합성 처리 539웰에서 두 지표별 공개 VC 최선 모델을 각 재표집에서 다시 고르는 계산과 parent 및 배양 단위 10,000회 재표집을 검증했다. 독립 실측 정확도는 미측정이다. 실측 채점에는 기존 동결 인자에 `--native`, `--native-pin`, `--native-protocol`을 더해 채점한다. [채점 코드](../../src/r19_plan/hcc_off_position_pilot_native_score.py) · [합성 영수증](data/hcc_off_position_pilot/native/score_smoke/receipt.json) · [지원 검산](data/hcc_off_position_pilot/native/score_smoke/support.tsv)

## 독립 확인 규모 산정

10 µM·24 h의 JHH-5·JHH-7 개발 구간으로 96군 배치의 정밀도를 점검했다. 개발 효과와 배양 표준편차 0.25를 가정한 두 지표 공동 통과확률 하한은 한 세포주에서 58–87%였다. 파일럿 자체가 제공해야 할 정보는 독립 배양 변동, 다른 위치 반복의 재현성, 공개 VC 대비 오차 차이다. 파일럿과 scaffold가 다른 정식 확인 후보 상한은 등록 단계에서 176군이다. [10 µM 규모 산정](HCC_OFF_POSITION_POWER_RESULTS.md)

## 실측 파일 인수 형식

| 파일 | 필요한 내용 |
|---|---|
| `wells.tsv` | `plate_slot`, `well`, `specimen_id`, `actual_cell`, `plate_id`, `culture_id`, `canonical_parent`, `material_lot_id`, `actual_uM`, `actual_hours`, `qc_status`, `qc_reason`. 예정된 좌표 768개를 한 줄씩 기록한다. DMSO의 `canonical_parent`와 `actual_uM`은 빈 값이다. `FAIL`에는 이유를 쓴다. |
| `cultures.tsv` | `cell`, `plate_id`, `culture_id`, `batch_id`, `independent_start_id`, `independence_record`. 16개 실제 plate와 배양의 연결 및 독립 시작의 근거를 기록한다. |
| `qc_record` | 실제 plate와 배양 기록의 검토, 사용한 약물 lot·노출, 발현 스케일 정합 및 사전 QC 규칙을 기술한 실험 기록이다. |
| `assay_receipt.json` | `mode`, `study_id`, `platform`, `target_scale`, `verified_by`, `wells_sha256`, `cultures_sha256`, `qc_record_sha256`, `measurements_sha256`. 실제 모드는 `measured`이며 등록 스케일 문자열은 `CMap2020_Level3_log2`다. |
| `measurements.npz` | QC 통과 표본 전체의 고유 `specimen_id`, 978개 고유 `genes`, 표본×유전자 순의 유한 실수 `values`. 두 축은 표본·유전자 ID로 맞춘다. |

G1의 원 예측 폴더를 `--forecast`에, 저장소의 동결 영수증을 `--forecast-pin`에 연결한다. `lock` 출력은 인수 명부와 지원을 기록한다. `score`에는 이 디렉터리와 `receipt.json`의 SHA-256, 공통 인수 인자 및 실측 파일을 제공한다. 합성 검산에만 `--synthetic`을 쓴다. [코드](../../src/r19_plan/hcc_off_position_pilot_score.py) · [원 예측 동결](data/hcc_off_position_pilot/forecast/freeze.json)
