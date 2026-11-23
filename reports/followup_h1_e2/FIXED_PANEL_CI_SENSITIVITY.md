**질문** — 후보 약물을 고정하면 LIMORE·LICOB의 추천 신뢰구간과 기존 판정이 달라지는가?<br>
**실제 바뀐 것** — 원 동결 예측과 점추정량을 재현하고, 독립 계통 또는 공여자만 2,000회 재표집했다.<br>
**주 비교 결과·구간** — LIMORE 구조 커널 대 약물 평균 ΔNDCG는 교차 +0.0295 [−0.0445, +0.0705], 고정 +0.0295 [+0.0099, +0.0490]였다.<br>
**판정** — LIMORE 공동 기준 2개가 사후 민감도에서 바뀌었다. LICOB K=2/4/8은 모두 미통과를 유지했다.<br>
**다음 결정** — 기존 확증 판정은 보존하고 고정 패널 조건에서 달라지는 불확실성을 함께 보고한다.<br>

# P0 고정 패널 사후 민감도

원 커밋은 **1912b5d**다. 동결 예측, 원 라벨, LICOB 보정/평가 분할과 원 집계 영수증을 해시 대조했다. 원 metrics.py와 fewshot.py로 원 점추정량과 교차 bootstrap을 재현했다. 수치 최대 오차는 5×10⁻¹¹ 미만으로 원 TSV의 반올림 범위였다. 학습 및 예측 재생성은 0회였다. [재현 영수증](p0_replay_receipt.json)

LIMORE는 독립 50계통과 exact parent 68개다. 원 구간은 계통×약물 2,400회 교차 재표집이었다. LICOB는 독립 29공여자와 exact parent 53개이며 H1의 38공여자와 분석 로스터가 다르다. K=2·4·8에서 평가 공여자는 반복별 27·25·21명이다. 보정된 100개 반복 예측을 각각 재사용했다. 원 구간은 공여자×약물 500회 교차 재표집이다. 고정 패널에서는 원 cohort에서 뽑은 공여자 가중치를 모든 K·반복·모델에 동일 적용했다. 각 반복의 원래 평가 멤버십을 유지한 뒤 유효한 반복 평균을 동등 가중했다. 점추정량과 기대 동률은 동일하다.

아래는 LIMORE의 주요 약물 평균 대조와 LICOB의 추천 대조 전체다. 구간의 0 포함 여부나 부호는 NDCG 28개, Precision 15개에서 바뀌었으며 LICOB는 두 지표를 합쳐 6개가 바뀌었다. 공동 판정은 별도로 확인했다. [전체 104개 대조](FIXED_PANEL_CI_SENSITIVITY.tsv)

| 코호트 | K | 모델 | 대조 | 지표 | 원 교차 Δ [95%] | 고정 패널 Δ [95%] | 0 기준 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LIMORE | — | structured | drug_mean | ndcg_at_5 | +0.0295 [−0.0445, +0.0705] | +0.0295 [+0.0099, +0.0490] | 변경 |
| LIMORE | — | structured | drug_mean | precision_at_5 | +0.0200 [−0.0608, +0.0640] | +0.0200 [−0.0040, +0.0440] | 유지 |
| LIMORE | — | ridge_structured_50_50 | drug_mean | ndcg_at_5 | +0.0254 [−0.0523, +0.0831] | +0.0254 [+0.0038, +0.0474] | 변경 |
| LIMORE | — | ridge_structured_50_50 | drug_mean | precision_at_5 | +0.0200 [−0.0707, +0.0720] | +0.0200 [−0.0040, +0.0440] | 유지 |
| LIMORE | — | pairwise_top5 | drug_mean | ndcg_at_5 | +0.0580 [−0.0467, +0.1288] | +0.0580 [+0.0270, +0.0912] | 변경 |
| LIMORE | — | pairwise_top5 | drug_mean | precision_at_5 | +0.0400 [−0.0600, +0.1280] | +0.0400 [+0.0080, +0.0760] | 변경 |
| LIMORE | — | ecfp4_path_elasticnet | drug_mean | ndcg_at_5 | +0.0378 [−0.0710, +0.1163] | +0.0378 [+0.0236, +0.0543] | 변경 |
| LIMORE | — | ecfp4_path_elasticnet | drug_mean | precision_at_5 | +0.0760 [−0.0640, +0.1360] | +0.0760 [+0.0480, +0.1040] | 변경 |
| LICOB | 2 | d_eb_structured | b_local_mean | ndcg_at_5 | −0.0517 [−0.1210, +0.0069] | −0.0517 [−0.0793, −0.0234] | 변경 |
| LICOB | 2 | d_eb_structured | b_local_mean | precision_at_5 | −0.0419 [−0.1167, +0.0388] | −0.0419 [−0.0781, −0.0055] | 변경 |
| LICOB | 2 | d_eb_structured | c_eb_shrinkage | ndcg_at_5 | +0.0065 [−0.0267, +0.0303] | +0.0065 [−0.0050, +0.0179] | 유지 |
| LICOB | 2 | d_eb_structured | c_eb_shrinkage | precision_at_5 | +0.0032 [−0.0402, +0.0247] | +0.0032 [−0.0135, +0.0181] | 유지 |
| LICOB | 2 | d_eb_pairwise_rank | b_local_mean | ndcg_at_5 | −0.0245 [−0.0762, +0.0223] | −0.0245 [−0.0551, +0.0070] | 유지 |
| LICOB | 2 | d_eb_pairwise_rank | b_local_mean | precision_at_5 | −0.0201 [−0.0951, +0.0603] | −0.0201 [−0.0585, +0.0210] | 유지 |
| LICOB | 2 | d_eb_pairwise_rank | c_eb_shrinkage | ndcg_at_5 | +0.0337 [−0.0277, +0.0829] | +0.0337 [+0.0147, +0.0534] | 변경 |
| LICOB | 2 | d_eb_pairwise_rank | c_eb_shrinkage | precision_at_5 | +0.0250 [−0.0354, +0.0646] | +0.0250 [+0.0062, +0.0447] | 변경 |
| LICOB | 4 | d_eb_structured | b_local_mean | ndcg_at_5 | −0.0249 [−0.0782, +0.0101] | −0.0249 [−0.0431, −0.0073] | 변경 |
| LICOB | 4 | d_eb_structured | b_local_mean | precision_at_5 | −0.0125 [−0.0793, +0.0430] | −0.0125 [−0.0411, +0.0146] | 유지 |
| LICOB | 4 | d_eb_structured | c_eb_shrinkage | ndcg_at_5 | +0.0014 [−0.0224, +0.0258] | +0.0014 [−0.0082, +0.0100] | 유지 |
| LICOB | 4 | d_eb_structured | c_eb_shrinkage | precision_at_5 | −0.0058 [−0.0385, +0.0284] | −0.0058 [−0.0221, +0.0077] | 유지 |
| LICOB | 4 | d_eb_pairwise_rank | b_local_mean | ndcg_at_5 | −0.0037 [−0.0453, +0.0344] | −0.0037 [−0.0273, +0.0206] | 유지 |
| LICOB | 4 | d_eb_pairwise_rank | b_local_mean | precision_at_5 | +0.0039 [−0.0736, +0.0803] | +0.0039 [−0.0301, +0.0408] | 유지 |
| LICOB | 4 | d_eb_pairwise_rank | c_eb_shrinkage | ndcg_at_5 | +0.0226 [−0.0223, +0.0809] | +0.0226 [+0.0034, +0.0406] | 변경 |
| LICOB | 4 | d_eb_pairwise_rank | c_eb_shrinkage | precision_at_5 | +0.0106 [−0.0376, +0.0757] | +0.0106 [−0.0108, +0.0329] | 유지 |
| LICOB | 8 | d_eb_structured | b_local_mean | ndcg_at_5 | −0.0055 [−0.0309, +0.0113] | −0.0055 [−0.0157, +0.0037] | 유지 |
| LICOB | 8 | d_eb_structured | b_local_mean | precision_at_5 | +0.0046 [−0.0484, +0.0384] | +0.0046 [−0.0152, +0.0232] | 유지 |
| LICOB | 8 | d_eb_structured | c_eb_shrinkage | ndcg_at_5 | −0.0016 [−0.0199, +0.0155] | −0.0016 [−0.0091, +0.0053] | 유지 |
| LICOB | 8 | d_eb_structured | c_eb_shrinkage | precision_at_5 | +0.0015 [−0.0322, +0.0316] | +0.0015 [−0.0123, +0.0134] | 유지 |
| LICOB | 8 | d_eb_pairwise_rank | b_local_mean | ndcg_at_5 | +0.0027 [−0.0302, +0.0352] | +0.0027 [−0.0132, +0.0203] | 유지 |
| LICOB | 8 | d_eb_pairwise_rank | b_local_mean | precision_at_5 | +0.0154 [−0.0639, +0.0759] | +0.0154 [−0.0178, +0.0522] | 유지 |
| LICOB | 8 | d_eb_pairwise_rank | c_eb_shrinkage | ndcg_at_5 | +0.0066 [−0.0228, +0.0474] | +0.0066 [−0.0078, +0.0211] | 유지 |
| LICOB | 8 | d_eb_pairwise_rank | c_eb_shrinkage | precision_at_5 | +0.0124 [−0.0494, +0.0763] | +0.0124 [−0.0168, +0.0428] | 유지 |

## 원 공동 판정과 고정 패널을 대입한 판정

| 코호트·설계 | 모델 | 원 공동 판정 | 고정 패널 사후 공동 판정 |
| --- | --- | --- | --- |
| LICOB/exact68/K2 | d_eb_structured | 미통과 | 미통과 |
| LICOB/exact68/K2 | d_eb_pairwise_rank | 미통과 | 미통과 |
| LICOB/exact68/K4 | d_eb_structured | 미통과 | 미통과 |
| LICOB/exact68/K4 | d_eb_pairwise_rank | 미통과 | 미통과 |
| LICOB/exact68/K8 | d_eb_structured | 미통과 | 미통과 |
| LICOB/exact68/K8 | d_eb_pairwise_rank | 미통과 | 미통과 |
| LIMORE_adult_HCC_cell_cold | drug_mean | 미통과 | 미통과 |
| LIMORE_adult_HCC_cell_cold | additive | 미통과 | 미통과 |
| LIMORE_adult_HCC_cell_cold | drug_ridge | 미통과 | 미통과 |
| LIMORE_adult_HCC_cell_cold | cellhit | 미통과 | 미통과 |
| LIMORE_adult_HCC_cell_cold | structured | 미통과 | 통과 |
| LIMORE_adult_HCC_cell_cold | ridge_structured_50_50 | 미통과 | 통과 |
| LIMORE_adult_HCC_cell_cold | pairwise_rank | 미통과 | 미통과 |
| LIMORE_adult_HCC_cell_cold | pairwise_top5 | 미통과 | 미통과 |
| LIMORE_adult_HCC_cell_cold | ecfp4_path_elasticnet | 미통과 | 미통과 |

LIMORE의 정정 구조 커널 및 Ridge+구조 커널 50:50은 NDCG 하한을 고정 패널 구간으로 바꾸면 사후 공동 기준에 든다. 약물 내 ρ, 이중 중심화 상호작용 ρ, Top-5 다양성, source OOF의 약물 평균·가산 대조에는 원 근거를 적용했다. Precision은 참고 축이다. LICOB의 공동 기준은 NDCG와 상호작용이 현지 평균 및 EB 수축 모두를 넘어야 한다. 고정 패널 구간을 적용해도 LICOB의 6개 모델×K 조합에는 공동 통과가 없었다.

고정 패널 구간은 등록 약물 위에서 계통·공여자 불확실성을 추정한다. 원 교차 구간은 후보 약물이 변할 때의 불확실성까지 포함한다. 두 추정량의 적용 범위를 함께 보고하며 기존 계약과 확증 판정을 보존한다.
