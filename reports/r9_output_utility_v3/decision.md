# P0-R9 decision

- **상태: 완료** (출력·보류 지표 정정). 성능 향상을 뜻하지 않는다.
- 정정 내용: 무출력 행의 output_error를 0 → NA로 바로잡고, 행 단위 출력 수와 donor-macro coverage/조건부 오류(유효 donor·seed 수 병기), 보조 합산 오류율, 25 donor 전체 coverage 표를 추가했다(`errata.md`). 모델·calibration·임계값·AUPRC는 변하지 않았다(검사 C5–C7).
- **사용 범위 판정(LR_donor, Corrected, 개발 진단 자료)**: nat·sparse_Endothelial·sparse_Fibroblast는 plan §5.4 1행(출력이 매우 적음 → 전면 자동 분류용 채택 보류), sparse_B는 3행(그 조건만 기술; 출력 대부분 parent 수준, coverage CI가 0.50 포함). 2행(조건부 오류 높음)은 확립된 조건 없음. 상세 `usability_decision.md`.
- 다음 행동: 현재 정책·모델 유지, 추가 실험 없음. 운영 요구사항이 정해지면 다음 독립 평가 전에 기준을 고정.
- 사용자 결정 필요: 용도 — 대부분의 세포에 자동 답변이 필요한지, 일부 고신뢰(주로 parent 수준 포함) 답변이면 충분한지.
