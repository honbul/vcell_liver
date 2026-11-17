# 연구 R5 — 결정 기록 (DECISIONS)

형식: 관측 → 가설 → 수정 하나 → 예상 차이 → 실제 결과 → 다음 결정. 규칙 파일은 `configs/perturbation_design_r5/protocol.yaml`.
각 항목은 해당 실행 **전에** 작성하고, 결과는 실행 후 같은 항목 아래에 덧붙인다.

## 0. 운영 기록

| 날짜 | 내용 |
|---|---|
| 2026-11-17 | 브랜치 `research/noise-aware-perturbation-design-r5`를 연구 B commit `d67bf68`에서 생성. 계획서 묶음을 `docs/hcc_perturbation_r4_r5/`에 그대로 복사(MANIFEST sha256 일치). |
| 2026-11-17 | 프로토콜 commit `a55bdc2` (발현 값을 읽기 전). |
| 2026-11-17 | 첫 build job 25334는 프로토콜 YAML 구문 오류로 **프로토콜 읽기 단계에서** 멈춤(값을 읽기 전). 블록 인용만 고치고(규칙 변경 없음) job 25335로 재실행. |
| 2026-11-17 | build job 25335: 원자료 경로 오타(`assets/raw` → `assets/raw/cmap2020`)로 instinfo 읽기 전 실패. 경로만 고쳐 재실행. |
