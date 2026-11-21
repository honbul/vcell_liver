# 동일 약물 RNA 전이에서의 기저 맥락 DEV 후속 검증

기존 X.A와 X3 결과를 본 뒤 등록한 X.B 후속 비교다. 원 X의 GSE92742 실측 10 µM·6 h, 공통 493 canonical parent, 다섯 세포주와 14,898 물리 처리 웰을 그대로 사용한다. 원래 다섯 held-cell, 세 held-parent fold, 세 DMSO 순환, 안쪽 select 세포주와 parent fold, 978 landmark를 유지한다. R18 및 그 외 보호 측정에 접근하지 않는다. 원 DEV Holm 판정은 원 발표의 8개 슬롯 파일에 그대로 보존한다. 새 X.B의 p값은 후속 탐색의 비보정 주 비교로 보고한다.

- **공통 B anchor:** 대상 세포와 같은 약물의 A375·A549·HT29·MCF7 실측 가운데 예측 대상 세포를 제외한다. 추론에는 나머지 비간계열 donor 3개 또는 4개만 들어간다. donor는 자신의 물리 RNA plate의 fit DMSO를 빼고 먼저 세포주 내 웰을 평균한 뒤 세포주 균등 가중한다. Outer-held parent의 target 처리·score DMSO는 예측에 없고, 등록된 다른 세포주의 조회 약물 반응만 donor 입력으로 접근한다. 내부 선택은 원 X의 독립 select DMSO를 사용한다. 이 명부는 기존 X B처럼 held target을 한 번에 하나로 제한한다. 다른 조각의 측정이 이 조각에 적합 label로 들어오는지 canonical parent와 role로 검사한다.
- 원 X의 inner 학습 세포에서만 같은 약물 donor를 모아 select용 anchor를 만든다. R17 `GAINS`의 여섯 값 중 하나를 select 공동 선택 손실로 고르고 모든 arm에 공유한다. 적합 레이블도 gain을 곱한 동일 약물 donor 평균을 빼고, 그 레이블이 속한 세포 자체와 간계열은 donor에서 제외한다. 적합 source에는 같은 약물의 다른 비간계열 세포를 적어도 하나 요구한다.
- `X1_B`, `X2_B`는 같은 anchor 위의 동일 FiLM 잔차 모델이다. 유일한 주 비교 변경은 X1의 fit-cell ID/OOV 균등 평균과 X2의 DepMap 기저 RNA/fit-only 표준화·PCA다. 원 X의 전체 gene head, 폭, parent 균등 MSE, seed, 두 학습률·다섯 epoch와 여섯 gain, 공동 checkpoint×gain 선택을 유지한다. 검증 손실은 `anchor + gain × residual`이다. `B_cell` 단독, `X2_B_context_off`와 fit-only 기저 profile 순열 `X2_B_context_shuffle`도 같은 규약으로 다시 적합한다.
- 주 비교는 다섯 세포주 전체의 `X2_B−X1_B` parent-macro RMSE·pooled R²다. HEPG2와 다른 세포주 각각, X2 대 anchor 및 두 대조, 동일 parent의 세포주 평균을 뺀 상호작용, fold와 plate 삭제 민감도를 별도로 보고한다. 10,000회 대응 parent 99% 구간과 방향성 공동 교환 p를 원 X 코드로 계산한다. 기존 실용 문턱 0.5% RMSE, 0.5%p R², 같은 방향 fold 2개 및 음성 대조도 그대로 적용한다. HEPG2는 간모세포종 계열이고 이 결과는 성인 HCC 외부 확인과 분리한다.

순서는 별도 donor 원명부 추출 → 예측 → 해시 동결 → 별도 채점이다. 모든 arm은 같은 처리 웰과 유전자, donor와 fit/select/score 역할을 사용한다.
