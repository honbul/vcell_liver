# R8 v2 — 미실행 기록 (NOT_RUN)

**작성일:** 2026-11-19 · 아래 항목에는 결과 숫자가 **없다.** 빈 칸을 성공·실패로 읽지 않는다.

| 항목 (비공개 기획서 2026-11-19 위치) | 상태 | 실행하지 않은 이유 | 실행 조건 (unlock) |
|---|---|---|---|
| R8-E3 support 다양성 대 반복 (§5.4) | 미실행 | 진입 조건은 "실제 다음 측정(support 획득) 결정이 있을 때"다. 이 트랙에는 예정된 획득 계획, 비용 단위(unique drug / plate row / treated well / support plate / control well), 예산 상한이 없다. v1 §10의 D16−REP8 비교는 상수 선택 방식에 따라 부호가 바뀌었고 centred C1은 동등했으므로, 같은 질문을 같은 방식으로 다시 돌리지 않았다 | 실제 support 측정 계획이 정해지고 그 비용 단위와 상한이 고정되면: 설계별·예산별 source inner 재선택, row/반복·shared plate 정밀도 보정 C0/C1, 고정 query manifest, 비용–성능 Pareto 표(`budget_retuned_precision_matched.tsv`) |
| query control 미관측 계약 (§5.2 입력 계약 4행) | 미실행 | 이 트랙의 사용 시나리오(HCC 기존 분할, source matched plate)에서는 query의 DMSO 대조군이 같은 plate에서 항상 측정된다. 기획서는 "실제 사용 시나리오가 요구할 때"만 별도 실험으로 둔다 | query plate의 대조군이 없는 사용 시나리오가 정의되면: 모든 방법에서 query-control block(ctrl kernel의 query 측)을 제거하거나 사전 고정한 source/support-only 대체를 동일하게 적용 |
| label-free overlap prior (missing 계약의 별도 후보, §5.2) | 미실행 | 기획서에서 "필요하면" 두는 선택 후보다. missing 계약은 0-overlap fallback과 source 재선택 두 가지로 비교했다 | missing 계약이 실제 주 시나리오가 되면 source plate 설계에서 label 없이 정한 overlap 기대값을 후보로 사전 등록 |
| 다른 compound plate·날짜·배양 shift stratum (§5.2) | 해당 자료 없음 | 적격 matched pair는 모두 한 compound plate의 X-tag 반복이다(`e1_report_inputs.json`의 eligibility). 같은 X 반복을 독립 날짜/배양이라 부르지 않는다 | 다른 compound plate/날짜/배양 반복이 확인되는 자료 |
| 독립 외부 HCC/반복 패널 (§5.2, §1.3) | 자료 없음 | 이 트랙에 새 외부 자료가 없다. 새 source 분할도 이미 노출된 자료의 내부 일반화 검사다 | 새 배양/날짜 HCC 패널(intervention·dose·batch·vehicle metadata와 보류 정답) |
| GPU 학습 | 0 GPU-시간 | 모든 패키지가 닫힌식 CPU 계산이다 (트랙 GPU 상한 0) | — |
