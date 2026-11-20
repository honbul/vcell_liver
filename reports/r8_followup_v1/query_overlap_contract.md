# R8 후속 분석 v1 — query 관측 환경 입력 계약 (query_overlap_contract)

**작성일:** 2026-11-18 · §8 분기(J1/J2/J3)에서만 쓰는 **입력 계약 확장**이다. 기존 `Episode.inputs(K)`에는 없던 항목을 별도 schema(`inp["o_qs"]`, `engine.inputs`)로 추가한다.

## 1. 추가되는 입력

`o_qS = W_q W_S'` (Q × K): query unit q와 support unit s의 detection plate 가중치 내적.
`W_u`는 unit u의 plate별 가중치 합이며, 가중치는 원 집계 규칙(조건 안에서 plate 균등 평균, unit 안에서 조건 균등 평균; `build_store.row_weights`)으로 정해진 **고정 집계 가중치**다. 각 unit에서 합이 1이다.

| 허용 | 허용하지 않음 |
|---|---|
| 사전에 계획된 query의 plate 배치(어느 detection plate에서 측정될 예정인가) | query의 처리 후 발현, 실제 오차, 반응 크기 |
| 고정 집계 가중치 | 처리 후 QC로 정한 가중치 |
| support의 plate overlap·row weight (기존 계약) | query 정답 평균의 역주입 |

DMSO 대조군 발현 사용은 기존 계약 그대로이며, query의 대조군 발현은 같은 plate의 DMSO 평균이므로 예측 시점에 관측 가능하다.

## 2. 모드

- **planned plate mode:** 위 `o_qS` 사용 (J1, J2, J3).
- **unknown new environment mode:** `o_qS = 0`. query가 어느 support plate와도 겹치지 않는다고 두는 것이며, plate 효과가 없는 생물학적 정답을 얻었다는 뜻이 아니다 (`*_unknown_env` 행).
- matched-plate E1에서는 `o_qS = 1`, E2에서는 `0`이다(한 행씩, 한 plate).

## 3. 공정성

같은 metadata를 받는 단순 기준선 J1(`b_q = a·mean(E) + c·(o_qS E)/K`, a,c ∈ {0,.25,.5,.75,1}, a+c ≤ 1, source 선택)을 항상 함께 보고한다. joint 모형(J2/J3)은 공통 보정 b_S를 **대체**하며 그 위에 더하지 않는다. 공통 τ_b, 환경 v_s·W W', 약물 τ_k·K, 대각 σ0²I + v_e·diag(q)는 하나의 PSD joint 행렬에서 SS/QS 블록을 꺼낸다(부록 C 7 검사 통과). v_e, v_s는 fold 학습 family의 plate 쌍 추정값으로 고정하고, τ_b/u, σ0²/u, τ_k/u만 source inner-validation에서 고른다(u = fold의 V3 v_tot).

## 4. HCC 적용

HCC query의 plate 배치는 기존 동결 분할의 query 목록과 target 메타데이터(`det_plate`)에서 같은 가중치 규칙으로 계산한다(처리 후 값 미사용). JHH7은 plate 1장이라 모든 query가 support plate를 공유한다.
