# R8 v2 — 정정(P0)과 관측 환경 전이 후속(E1/E2-A) 결과

**작성일:** 2026-11-19 · 브랜치 `research/r8-observation-transfer-v2` (기준 R8 후속 v1 tip `2f4329e`) · 사양 동결 **F1 `1b59b76`** (YAML 문법만 고친 `6a66d0b`) · source 결정 동결 **F2** `__F2__` · 근거: 비공개 기획서(2026-11-19) §1, §2, §5, §8, §9.

모든 수치는 CMap LINCS 2020 Level 3, 978 landmark, (세포주, 약물 그룹) unit 단위 ΔRMSE 또는 centred RMSE다. 차이는 "앞 − 뒤"이고 음수면 앞 방법의 오차가 작다. source는 family cluster bootstrap 95% CI(2,000회), HCC는 약물 ID 동기화 bootstrap. `[k/3]`은 앞 방법이 좋았던 outer fold 수.

**노출 지위.** 이 트랙의 모든 자료는 R8 또는 R8 후속 v1에서 이미 분석되었다. P0는 **기존 결과 정정**, E1/E2-A의 source 수치는 **이미 노출된 자료의 후속 분석**(새 matched episode 집합이지만 같은 family·plate), HCC 표는 **동결된 후속 평가**다. 새 독립 자료나 blind test는 없다.

__BODY__
