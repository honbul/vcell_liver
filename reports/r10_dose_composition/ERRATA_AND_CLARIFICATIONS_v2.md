# R10 v1 결과의 상태·계약 정정 및 명확화 (v2, 2026-11-19)

> **정정 노트.** 이 문서는 R10 v1 문서(`RESULTS.md`, `DECISIONS.md`, `CLAIMS_AND_LIMITATIONS.md`, `README.md`, `BEGINNER_SUMMARY.md`; 기준 commit `0c35756`)의 해석 범위를 바로잡는다. v1 수치·표·run은 바꾸지 않고 그대로 보존한다. 근거: private plan 2026-11-19 §7 도입부, §7.1.3, §7.3.2 끝. 후속 측정은 [`contract_matrix_v2/RESULTS.md`](contract_matrix_v2/RESULTS.md).

## E-1. test 평가 범위: test에서 평가된 learned arm은 선택된 D11 하나뿐이다

- [관측] v1 test 평가(`r10_test_v1`, freeze `f0ded27`, job 26427)는 **D11(cross 학습 head, seed 0·1·2)** 과 비학습 기준선(control, nearest_low/high, mixture=D00, global_moment, state_moment, ot_displacement, oracle_replicate)만 채점했다. D10·D01·W, same 학습 head(C2)는 test에서 채점하지 않았다. mixture_t(t*=0.325)는 freeze 이후 C3에서 test에 추가 채점했다(사후 비교, blind 아님).
- [정정] 따라서 v1 test 결과는 "development 규칙으로 선택한 D11 v1이 D00 등 비학습 기준선보다 나았다"는 **선택 모델 하나의 확인**이다. RESULTS §4의 2×2 head별 효과(D10/D01/상호작용)와 W 비교는 development(head 선택 집합)에서만 측정됐으며, test가 head별 factorial 효과를 확인한 것처럼 읽지 않는다.
- v2에서 D10·D01·W(cross)와 same 학습 arm의 test 수치를 처음 계산하지만, test 38약물은 이미 노출된 자료이므로 **"이전에 노출된 자료의 동결된 후속 비교"** 로만 표시한다(blind/독립 test 아님).

## E-2. C3의 t*는 train-cross에서 적합했고 v1 freeze 이후 추가한 비교다

- [관측] `r10_c3_tstar_v1.json`은 train 약물의 **cross** triplet에서만 t*를 적합했다(E100·E1000 모두 0.325). C3는 freeze(`f0ded27`) 이후에 만든 기준선이다.
- [정정] RESULTS §0·§6과 DECISIONS C3의 same 계약 비교(예: "C2 D10 − mixture_t(same) = −0.002 [−0.011, +0.005]")에 쓰인 mixture_t는 **same 계약에 맞춘 t*가 아니다.** 그러므로 "same 계약에서 learned π-head가 (same에 최적인) 전역 mixing weight와 같은 수준"이라는 문장은 계약이 맞지 않는 기준선과의 비교다. v2에서 train-same t*(`tstar_same.json`: E100 0.35, E1000 0.40)를 따로 적합해 두 t*를 모든 칸에 함께 제시한다.

## E-3. "same" 계약은 같은 replicate 라벨일 뿐 plate 효과를 제거한 조건이 아니다

- [관측] 적격 triplet 전체(train·development·test, `eligible_triplets.tsv`의 plate 열로 계산): **same** triplet 2,057개 모두 low·high endpoint가 서로 다른 plate이고(`endpoints_same_plate=False`), query는 low endpoint plate(1,024) 또는 high endpoint plate(1,033) 중 하나와 같다. **cross** triplet 2,076개는 모두 query가 두 endpoint 어느 plate와도 다르다.
- [정정] `query_same_plate_as_endpoints=False`는 "query·low·high 셋이 모두 같은 plate는 아니다"라는 뜻이고, "어느 endpoint와도 plate가 다르다"는 뜻이 아니다. same 계약은 **동일 replicate 라벨(같은 plate 묶음)** 의 계약이며 plate 효과가 제거된 실험이 아니다. RESULTS·DECISIONS의 "replicate 이동이 없는 계약"이라는 표현은 "replicate 라벨이 같은 계약(plate 차이는 남음)"으로 읽어야 한다.

## E-4. CI가 0을 포함하는 D10 − mixture_t 비교는 동등성의 증거가 아니다

- [관측] v1에서 same 계약 D10 − mixture_t의 구성비 TV 차이 구간은 0을 포함했다(C1 head +0.005 [−0.005, +0.014], C2 head −0.002 [−0.011, +0.005], development). 동등성 margin은 사전에 정하지 않았다.
- [정정] 이 결과는 "차이 미확립"이다. RESULTS §0 세 번째 항목의 "구분되지 않았다", §6 추론과 DECISIONS C3의 "같은 수준" 표현은 **동등성 입증으로 읽지 않는다.** 또한 이 비교는 구성비 TV에 대한 것이므로, C2의 Q-head·ED 개선 전체를 전역 mixing weight가 설명했다는 뜻도 아니다. v2는 사전 margin(actual TV ±0.005, 90% 구간)을 결과 보기 전에 고정해 동등성을 별도로 판정한다(`configs/r10/contract_matrix_v2.json`).
- [정정] 기존 0.57 비율(C2 이득/C1 이득)은 두 계약의 이득 크기 비율일 뿐이며 "생물학 57% / batch 43%" 같은 원인 분해로 바꾸지 않는다.

## E-5. 조립 불일치(declared vs reread)는 채점 오류가 아니다

- [관측] v1 채점은 생성 cloud를 고정 atlas로 다시 읽은 구성비(`comp_tv`)로 했고, head가 선언한 π와 재판독 π의 차이(D11 test E1000 평균 TV ≈ 0.016)는 별도 열(`assembly_tv_head_vs_reread`)로 보고했다.
- [명확화] 이 차이는 학습 목적(선언 π의 KL)과 조립 표현(soft membership 가중·Q 이동) 사이의 불일치이며, 선언 π로 점수를 매긴 채점 버그가 아니다. v1 수치는 소급 변경하지 않는다. 0.016 불일치와 0.020 성능 개선은 서로 다른 비교이므로 나누어 비율로 해석하지 않는다. 진단은 v2 `assembly_audit_v2/`.
