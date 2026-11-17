# 연구 R8 — 실행 규약 (PROTOCOL)

**작성일:** 2026-11-17 · 기계가 읽는 규약은 `configs/replicate_calibrated_r8/protocol.yaml`이고, 이 문서는 사람이 읽는 요약이다. 둘이 다르면 yaml이 기준이다.

## 0. 읽은 계획 문서 (내용은 저장소에 넣지 않는다. 해시만 기록)

| 문서 | bytes | sha256 | 읽음 |
|---|---|---|---|
| 00_OVERVIEW_AND_GIT_REVIEW.md | 16097 | eb5659832a23a4fa5e6d87dbf5fbc4c5d4ceb0553def6fca8d869fdac20be4f9 | 예 |
| 03_R8_REPLICATE_CALIBRATED_ADAPTATION.md | 19146 | 3f7ea79bd875748188340cf4ae7029af0d786cc1f1c349f37f1f3ab14913be6c | 예 |
| 04_AGENT_GOALS.md | 8266 | 17928d34375b4ea61064fa5ba1d7c000d2e2b8bf476cf5e22e1da3f98336f8ea | 예 |
| 05_EVIDENCE_AND_REFERENCES.md | 18312 | a157014dfb39d50ef91f53547fe44beae2e749bf553fd90156e4338bc1fb3fb9 | 예 |
| 01_… | 17683 | 5a139adf… | 아니오(다른 연구용) |
| 02_… | 20081 | 51a86118… | 아니오(다른 연구용) |

## 1. 질문

HCC 세포주에서 약물 16개의 처리 후 RNA만 볼 수 있을 때,

1. 약물별 잔차 보정(C1, C2)이 **공통 보정(C0)** 보다 실제로 나은가?
2. support 측정들의 **복제·plate 공유 구조**를 반영한 관측 공분산이 도움이 되는가? (B10 vs B00)
3. support-centring에 지워지지 않는 **전역 대조군 상태**가 도움이 되는가? (B01 vs B00)
4. 둘을 함께 쓰면? (B11)

## 2. 자료와 단위

- CMap LINCS 2020 Level 3, 10 µM, 24 h, landmark 978 유전자. 연구 B가 만든 배열을 그대로 읽는다(복사·재다운로드 없음).
- **한 행(unit)** = (세포주, 약물 그룹). 값 = 조건별·plate별 (처리 후 − 같은 plate 대조군 평균)의 동일 가중 평균 = ΔRNA.
- **독립 단위** = 채점은 약물 그룹, 요약은 source에서는 family, HCC에서는 계열별 약물 그룹 bootstrap.
- XC 혼합 pool 29개(4,572 unit)는 제외. 남은 source는 132 세포주·126 family·74,277 unit.
- source-train 121 세포주(115 family), 확인 11 세포주(11 family, 그중 unit ≥ 40인 것만 episode가 됨). HCC 3계열은 동결 전 잠금.

## 3. 방법 (모두 같은 K개 support, 같은 query)

| 이름 | 내용 |
|---|---|
| no_change | 0 |
| same_drug_mean | f0 (family 균형 같은 약물 평균) |
| shrunk_same_drug_mean | f* = f0·n/(n+κ_f) |
| mean_offset | f0 + support 평균 잔차 (R2의 기준선, 축소 없음) |
| **C0** | f* + b, b = 평균 잔차 × K_eff/(K_eff+κ_b) |
| **C1** | C0 + η·(학습 없는 kernel 회귀: Tanimoto + 대조군 32축 + f0 33축) |
| **C2 = B00** | C0 + η·(학습된 embedding의 kernel 회귀), 상태 없음, Ω = I |
| B01 | B00 + 전역 상태(FiLM, 약물 표현에 곱함) |
| B10 | B00 + 관측 공분산 Ω(복제·plate 공유) |
| B11 | 둘 다 |
| G_concat, G_dummy | 전역 상태를 더하기로 넣은 것 / 같은 크기의 무의미한 벡터를 넣은 것(절제) |
| generic CNP | 앵커·분해 없는 일반 conditional neural process |
| TranSiGen as-is / offset / full FT | XC를 빼고 다시 학습한 공개 모델과 그 미세조정 |

관측 공분산: Ω_ij = [δ_ij(v_r + v_e q_i) + v_s o_ij]/(v_r+v_e+v_s) + 0.05 δ_ij, 비대각은 0.9배. o_ij는 두 unit이 공유한 plate의 가중치 곱 합, q_i는 unit의 plate 가중치 제곱 합. 상수는 source-train 복제 plate 쌍에서만 추정. 풀이는 Cholesky(역행렬을 직접 만들지 않음).

## 4. 선택과 동결

1. 진단 D0/D1/D2 → 공분산 상수 고정.
2. 닫힌 식: κ_f, κ_b, C1의 λ·η를 source-train dev episode에서 선택.
3. 학습 모델: 3-fold family 교차검증으로 (학습률 ∈ {1e-4, 3e-4, 1e-3}, step ≤ 6000, 200 step마다)를 arm별로 선택. 기준은 K=16 family 동일 가중 ΔRMSE.
4. 고른 step 수로 source-train 전체에서 seed 42/43/44 학습. 확인 family에서 한 번 채점.
5. 개정은 최대 2회, 사전 기록 필수(DECISIONS.md).
6. 동결 커밋(설정 + 동결 manifest + 코드 SHA) → HCC 평가 1회(모델 seed 3 × 분할 seed 3, K=8/16/32).

## 5. 지표

- 주 지표: query 약물 그룹별 ΔRMSE의 평균(낮을수록 좋음). 보조: 약물 평균을 뺀 centred RMSE, Pearson, 방향 일치 등(`metrics.per_unit`).
- 차이는 같은 query에 대한 짝 차이와 bootstrap 95% 구간으로 보고한다(HCC: 약물 그룹 2000회, seed 20261117; source: family).
- 라벨 비용은 `support_cost.tsv`에 약물 그룹 수·처리 well 수·독립 plate 수·대조군 well 수를 따로 적는다.

## 6. 하지 않는 것

생성 모델, 지식 그래프, 능동 선택, 다른 플랫폼, 유전형 입력. 결과가 음성이어도 자동으로 확장하지 않는다.
