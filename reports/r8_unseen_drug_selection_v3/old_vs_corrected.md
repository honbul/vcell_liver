# R8 E2-B: 이전(v2) 결과와 분리 정정(v3) 결과

**작성일:** 2026-11-20 · 브랜치 `research/r8-unseen-drug-selection-v3` · freeze `f0589c6` · 근거: 비공개 기획서 2026-11-20 §4.

## 1. 정정 이유

v2 E2-B source-reference-unavailable 검증(`src/r8_transfer_v2/e2b.py` stage `sru`, 사양 F3 `84dbfa3`)은 drug fold j의 query 약물을 **추론 입력**(source f0 표, f0 PCA, kernel scale, ridge 사상)에서 제거했다. 그러나

1. **상수 선택**(κ_b, ridge α, λ, η)의 inner episode가 같은 drug fold j의 약물을 query로 사용했다. 즉 outer query 약물의 다른 family(inner-val family) 정답으로 상수를 골랐다. v3 P0 검사로 다시 세어 보면 block마다 outer query 약물 group **416–775개**가 v2 inner 선택의 query 정답에 들어 있었다(`p0/overlap_checks.tsv`, level `v2_contract_reference`).
2. **κ_f**는 같은 약물 참조가 있는 full-reference P0 선택(`p0sel_inner{k}_v1`, 모든 약물 사용)에서 가져왔다(κ_f = 4 / 8 / 8).

따라서 v2 결과는 **"예측 시 참조 부재 실험"**으로 보존한다(값과 문장은 v2 `RESULTS.md` §4.2 그대로, 수정하지 않음). v3는 같은 outer episode에서 선택까지 분리한 **"분리 정정 버전"**이다.

## 2. 무엇을 바꿨나

| 항목 | v2 (예측 시 참조 부재) | v3 (분리 정정) |
|---|---|---|
| outer query / support episode | E1 RNG, fold j query, 비보류 support | **동일** (9 block 모두 cell×seed별 query·support 수·적격 여부 일치) |
| 보류 범위 | fold j group | fold j group의 **alias closure** (6 cluster, block당 +2~5 group; query 집합은 동일) |
| inner 선택 query | **fold j (outer query 약물)** | outer 허용 약물 A를 3개 inner drug fold로 나눈 각 fold (outer query·alias와 교집합 0) |
| inner 적합(f0·PCA·scale·ridge) | fold j 제거 | fold j closure + 해당 inner fold closure 제거 |
| κ_f | full-reference P0 값 (4 / 8 / 8) | inner 계약에서 κ_b와 함께 재선택 (9 block 모두 32 = grid 상한) |
| ridge α, κ_b, λ, η | 누수 있는 inner에서 선택 | 새 inner 계약에서 재선택 (`p1/constants_v2_vs_v3.tsv`) |
| arm | fb/ridge × C0/C1chem + ref_C0/ref_C1, K 8/16/32 | 4 arm, K = 16만 |
| CI | family cluster bootstrap | **two-way family×drug paired bootstrap** (centring 재계산; family-only는 참고) |
| 주 대비·실용 기준 | 사전 라벨: centred CI < 0 | `ridge_C1chem − fb_C0` centred RMSE, 상대 ≤ −0.5%, CI 상한 < 0, fold ≥ 2/3 |

## 3. 수치 비교 (K = 16, 71 family, 같은 query 38,302개)

| arm | v2 centred | v3 centred | v2 ΔRMSE | v3 ΔRMSE |
|---|---:|---:|---:|---:|
| fb_C0 | 0.543523 | 0.543523 | 0.567692 | 0.565477 |
| fb_C1chem | 0.542200 | 0.542183 | 0.566230 | 0.564006 |
| ridge_C0 | 0.542443 | 0.542192 | 0.562691 | 0.562175 |
| ridge_C1chem | 0.541126 | 0.540864 | 0.561262 | 0.560795 |

fb_C0의 centred RMSE는 상수와 무관하다(query마다 같은 보정이므로 centring 후 예측 0). v3가 v2 값을 소수 7자리까지 재현한 것은 두 버전이 같은 query 집합을 채점했다는 확인이다.

| 대비 | v2 (family CI) | v3 (two-way CI) | v3 family-only CI (참고) |
|---|---|---|---|
| **ridge_C1chem − fb_C0 centred (주)** | −0.00240 [−0.00353, −0.00149], 상대 −0.44% | **−0.00266 [−0.00394, −0.00142], 상대 −0.489%**, fold 3/3 | [−0.00384, −0.00169] |
| ridge_C1chem − fb_C0 ΔRMSE | −0.00643 [−0.00852, −0.00452] | −0.00468 [−0.00726, −0.00240], 상대 −0.83% | [−0.00674, −0.00278] |
| ridge_C0 − fb_C0 centred | −0.00108 [−0.00154, −0.00063] | −0.00133 [−0.00215, −0.00059] | [−0.00181, −0.00088] |
| ridge_C1chem − ridge_C0 centred | −0.00132 [−0.00241, −0.00054] | −0.00133 [−0.00237, −0.00043] | [−0.00242, −0.00051] |
| fb_C1chem − fb_C0 centred | −0.00132 [−0.00241, −0.00054] | −0.00134 [−0.00244, −0.00041] | [−0.00245, −0.00050] |

## 4. 바뀐 주장

- **관찰.** 선택 단계의 누수를 제거해도 centred 개선의 방향과 크기는 유지되었다(−0.00240 → −0.00266). 정정으로 결과가 나빠지지 않았다.
- **관찰.** ΔRMSE 기준 개선은 −0.00643 → −0.00468로 작아졌다. 주로 fb_C0의 ΔRMSE가 0.56769 → 0.56548로 좋아졌기 때문이다(재선택된 κ_f = 32, κ_b).
- **관찰.** two-way CI는 family-only CI보다 넓다(약물 재표집 반영). 주 판정은 two-way CI로만 했다.
- **변경된 주장.** v2의 "chemistry_transfer_without_same_drug_response 충족"은 v2 사전 라벨(CI < 0만 요구)에 따른 것이었다. 이번 사이클의 새 사전 명세(상대 ≥ 0.5% 감소)로는 주 대비가 **−0.489%로 기준 미달 → "작은 개선, 채택 보류"**다. v2 수치를 같은 기준으로 다시 읽어도 −0.44%로 기준 미달이다(사후 비교, 판정에는 쓰지 않음).
- **유지되는 주장.** 같은 약물 source 반응을 쓰는 계약(ref)과의 큰 차이(v2: −0.028)는 이번에 다시 계산하지 않았다(ref arm은 이번 패키지 범위 밖). v2 수치는 v2 계약의 결과로만 인용한다.
- **해석 범위.** 같은 source family·약물을 반복 사용한 개발 검증이며, 외부 blind 결과가 아니다.
