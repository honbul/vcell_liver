# R10-E1a / E2 실행 명세 (freeze 전 작성, 2026-11-19)

기계 판독 명세: [`configs/r10/contract_matrix_v2.json`](../../../configs/r10/contract_matrix_v2.json). 사전 대비 목록: [`manifests/r10_dose_composition/contract_matrix_v2/predeclared_contrasts.tsv`](../../../manifests/r10_dose_composition/contract_matrix_v2/predeclared_contrasts.tsv). 입력 hash: [`artifact_registry.tsv`](../../../manifests/r10_dose_composition/contract_matrix_v2/artifact_registry.tsv). 근거: private plan 2026-11-19 §7.1–7.2.

이 문서를 포함한 commit이 **freeze commit**이다. 이 commit 이전에는 development·test 약물의 v2 평가를 한 번도 채점하지 않았다(파이프라인 점검은 train 약물 16 triplet/세포주의 pilot으로만 함).

## 노출 지위

- development 35약물: 모든 learned head의 early stopping·weight decay 선택 집합(낙관적).
- test 38약물: v1에서 D11[cross]·비학습 기준선·mixture_t[cross]가 이미 채점된 자료. v2 test 수치는 **이전에 노출된 자료의 동결된 후속 비교**이며 blind/독립 test가 아니다.

## 가설과 범위

cross 학습 head의 same 악화는 학습·평가 계약 불일치와 관련되며, same 학습 head는 same에 맞춘 단순 기준선(t*_same, W_same)과 비교해야 해석된다. 이 설계는 환경의 인과 기여율을 식별하지 않는다.

## 입력과 arm (모든 칸 공통)

| arm | 학습 계약 | 출처 |
|---|---|---|
| D00 | 없음(한 번만 채점, 중복 합산 금지) | empirical log-dose mixture |
| tstar[cross] | train cross | v1 `r10_c3_tstar_v1.json`, t*=0.325/0.325 (E100/E1000) |
| tstar[same] | train same | **신규** `tstar_same.json`, t*=0.35/0.40, 같은 격자 |
| D10/D01/D11[cross], W[cross] | cross | v1 `r10_b3_v1` seed 0·1·2 (hash 일치) |
| D10/D01/D11[same] | same | v1 `r10_c2_same_v1` seed 0·1·2 (hash 일치) |
| W[same] | same | **신규** `train_same_W` seed 0·1·2 |

W[same] 학습은 v1 W와 같은 코드 경로(`train.py --direction same --heads W`)·같은 격자·같은 선택 규칙(same development ED 최소, wd ∈ {1e-4, 1e-2})이다. 결과: 세 seed 모두 wd 1e-4 선택, 선택 epoch 15/20/15, dev ED 0.1265/0.1209/0.1216(초기 0.2250). π/Q(KL·잔차 MSE, 최대 600 epoch)와 W(ED, 최대 150 epoch)는 loss와 최적화 비용이 다르다.

## 집계·pairing

seed → triplet → replicate 방향 → 세포주 동일 가중 → 약물; 약물 bootstrap B=2000(seed 20261117). 칸 안의 비교는 두 arm이 모두 유한한 triplet 교집합에서 계산한다. 학습 계약 효과(X[same] − X[cross])는 같은 평가 계약의 같은 query well에서 비교한다. 평가 계약 간 비교는 두 계약이 모두 적격인 (세포주, 약물, episode) 공통 집합에서 약물 단위로만 짝짓는다.

## 주 대비 (test, E1000, actual 구성비 TV, a − b)

| id | a | b | 평가 계약 |
|---|---|---|---|
| P1 | D11[same] | D11[cross] | same |
| P2 | D11[cross] | D11[same] | cross |
| P3 | D10[same] | tstar[same] | same |
| P4 | D10[cross] | tstar[cross] | cross |
| P5 | D11[same] | W[same] | same |
| P6 | D11[cross] | W[cross] | cross |

- 다중성: 6개에 Holm(두쪽 percentile bootstrap p), Bonferroni 동시구간 병기.
- 실용 기준: |Δ actual TV| ≥ 0.005(기획서 제안값; 이 commit에서 고정).
- 동등성: 90% 구간이 (−0.005, +0.005) 안일 때만 "margin 내 동등". 95% 구간이 0을 포함하는 것만으로는 "차이 미확립".
- 판정 라벨: practical_difference_established / difference_below_practical_threshold / equivalent_within_margin / not_established.

보조(보정 없음, 모두 공개): 모든 칸의 X − D00, D10 − t*(일치/반대 계약), D11 − D10(상태 내부·ED), D11 − W(TV·ED 별도), W − t*, X[same] − X[cross], arm_interaction(D11 − D10 − D01 + D00), contract_interaction, E100, development, 세포주·replicate 방향·query plate 관계 층.

## E2 진단 (새 학습 없음)

같은 query에 대해 TV(π_declared, π_query), TV(π_actual, π_query)(주 지표), TV(π_declared, π_actual)를 학습 계약·평가 계약·arm·episode·세포주·약물별로 연결한다. D00은 |declared − actual| ≤ 1e-6이어야 한다. endpoint 진단(membership entropy, 상태별 effective n 최소, 저지지 상태 질량, π 이동 TV, Q 이동 RMS)의 삼분위 층별 민감도를 낸다. 불일치가 큰 약물로 주 분석을 다시 정의하지 않는다.

## 조건부 패키지의 진입 규칙 (development만 사용, test로 열지 않음)

- U1: matched 칸의 D10에서 약물 평균 (actual TV − declared TV) ≥ 0.005이고 95% 하한 > 0 — **두 학습 계약 모두**.
- U2: matched 칸에서 (D10, D00), (D11, D00), (D10, t*), (D11, t*) 중 하나라도 declared 기반과 actual 기반 추정의 부호가 다르고 두 구간 중 하나가 0을 제외.
- epoch 순위: v1에는 best checkpoint와 declared-TV 곡선만 있어 평가 불가로 기록.
- §7.2.3 목적함수 arm: U1 또는 U2일 때만. §7.2.4 hard partition: §7.2.3과 함께만. §7.1.5 source OOF(E1b): §7.2.3이 열릴 때만.

## 비용(pilot 실측)

same W seed 0(두 wd): wall 4분 19초, CPU 1,905초, 최대 RSS 0.84 GB(8 CPU). seed 1·2: 6분 31초 / 3분 51초. 평가 pilot(train 16 triplet/세포주): heads 묶음 task당 CPU ~75초. 전체 평가는 30 task(2 role × 3 세포주 × 5 묶음), 예상 ≈ 20 CPU-시간 이내로 24시간 제한 안.
