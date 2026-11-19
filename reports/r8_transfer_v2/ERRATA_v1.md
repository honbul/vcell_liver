# R8 후속 분석 v1 보고서 정정 (P0-A 문구 정정)

**작성일:** 2026-11-19 · 대상: `reports/r8_followup_v1/FINAL_RESULTS.md`, `BRANCH_DECISION.md`, `NOT_RUN.md` (브랜치 `research/r8-followup-observation-transfer-v1`, tip 2f4329e). 근거: 비공개 기획서(2026-11-19) §5.1 P0-A. v1 문서의 원문은 지우지 않았다. 아래 정정이 우선한다. 수치 근거는 `manifests/r8_transfer_v2/anchor_matched_rows.tsv`, `anchor_decomposition.tsv`, `reproduction_row_kinds.tsv`와 `RESULTS.md` §1.

| # | v1 문구 (위치) | 문제 | 정정 |
|---|---|---|---|
| 1 | "C0의 b는 모든 query에 같으므로 centred RMSE에서 정의상 사라진다(**C0 = 축소 평균 0.52182**)" (FINAL_RESULTS §2.1) | C0와 centred 지표가 같아지는 것은 **같은 κ_f의 anchor**일 때뿐이다. v1의 `shrunk_mean`은 κ_f를 따로 고른 baseline이라(final_inner: shrunk κ_f = 4, C0 κ_f = 8 at K16; inner1: 4 vs 8) 일반적으로 C0와 centred 값이 다르다 | "C0의 centred 예측은 같은 κ_f의 `anchor_matched` f*와 같다." C0/C1이 실제 쓴 κ_f로 만든 `anchor_matched` 행을 추가했다. `shrunk_mean → C0`는 서로 다른 anchor 선택을 포함하는 성능 비교로 따로 둔다 |
| 2 | "Ω는 거의 변하지 않았다(**예측 차이 ~1e-6**)" (FINAL_RESULTS §0), "정규화된 Ω와 예측이 1e-6 수준만 달라졌다" (BRANCH_DECISION), "Ω 차이가 1e-6 수준" (NOT_RUN) | 표의 값(V3 − V0 both, +0.0000022 등)은 **family 평균 ΔRMSE의 차이**다. Ω 행렬이나 개별 예측 벡터의 차이를 측정한 값이 아니다 | "V3와 V0의 family 평균 ΔRMSE 차이는 2×10⁻⁶ 수준이었다." Ω·예측 벡터 차이에 대한 주장은 철회한다(측정하지 않음) |
| 3 | "공분산의 효과는 **전부** 공통 보정의 축소량에서 나왔다" (FINAL_RESULTS §0), "거의 전부 공통 보정의 축소량(α) 경로" (BRANCH_DECISION) | 정확히 성립하는 곳은 **E1/E2의 C0 공통 보정**뿐이다(한 plate support라 Ω가 compound symmetric이고 GLS 가중이 정확히 균등: weights-only = 0). 일반 OOF에서는 축소량이 주 경로지만 가중의 조건부 기여가 남는다(both − shrink-only = +0.00127 − 0.00172 ≈ −0.00045; weights-only − C0 −0.00014 [−0.00040, +0.00018]) | "E1/E2의 C0 공통 보정에서 공분산 효과는 축소량 경로로 완전히 설명되고, 일반 OOF에서는 축소량이 주 경로이며 가중의 조건부 기여가 작게 남는다." 이 주장을 C1의 λΩ 정규화로 확대하지 않는다 |
| 4 | `v0_reproduction_existing_panels.tsv`의 105행을 "V0 재현"으로 한데 묶은 표기 | 원 상수 C0/C1 재현은 60행(fx_C0/C1 × iid/V0 × K 3개 × 확인 + HCC 4행)이고, `shrunk_mean` 15행은 κ_f를 **독립 재선택**한 값(4 vs 원 8)이라 재현이 아니다(차이 최대 0.0044) | `reproduction_row_kinds.tsv`에 `row_kind` 열을 붙여 구분했다 |
