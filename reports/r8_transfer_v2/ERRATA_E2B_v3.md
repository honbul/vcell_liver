# 정정 안내 — R8 v2 E2-B source-reference-unavailable (2026-11-20)

v2 `RESULTS.md` §4.2와 `NOT_RUN.md`의 원문은 그대로 보존한다. 이 파일은 정정 안내만 추가한다.

- v2 E2-B sru는 outer query 약물을 **추론 입력**에서는 제거했지만, **상수 선택**(κ_b, ridge α, λ, η)의 inner episode가 같은 drug fold의 약물을 다른 family(inner-val)에서 정답으로 사용했고, κ_f는 full-reference P0 선택값이었다.
- 따라서 v2 수치(예: ridge_C1chem − fb_C0 centred −0.00240 [−0.00353, −0.00149])는 **"예측 시 참조 부재 실험"**의 결과로만 읽는다. "같은 약물 반응을 전혀 보지 않은" 결과가 아니다.
- 선택까지 분리한 정정 결과는 브랜치 `research/r8-unseen-drug-selection-v3`의 `reports/r8_unseen_drug_selection_v3/` (freeze `f0589c6`): 같은 query에서 −0.00266 [−0.00394, −0.00142] (two-way family×drug CI), 상대 −0.489% → 이번 사이클 사전 기준(0.5%) 미달, **작은 개선, 채택 보류**.
