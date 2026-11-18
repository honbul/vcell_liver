# R9 RESULTS — 관측 / 추론 / 미검증 분리

모든 GSE149614 수치는 **개발 증거**(바깥 LOPO 10명; HCC02·HCC09 포함 전원 이전 노출)다. 통계 단위는 donor, 괄호는 donor bootstrap 95% CI. seed·회전은 donor 안에서 평균했다. 숫자 출처 파일을 각 절에 적었다.

## A0 — inventory (`label_inventory.tsv`, `donor_class_counts.tsv`, `rare_definition_by_fold.tsv`, `cohort_overlap.tsv`)

**관측**
- GSE149614: 71,915세포, 10명, 25 시료(부위 Tumor/Normal/PVTT/Lymph). 저자 label 6개 대분류; 세부 아형 이름 없음. `res.3` 군집 53개(모두 한 대분류에 100% 포함).
- 대분류 학습 fold 최소 비율 2.55%(Fibroblast) → 자연 희귀 대분류 없음.
- Tier 2 다환자 자연 희귀 군집: Myeloid c41·c46·c52(모든 fold ≤1%, 학습 donor ≥3), c44(2–3 donor), B c40. 환자 단독: B c50(HCC02), Myeloid c53 등.
- GSE151530 HCC: 25명, 32 시료, 50,023세포; 대응 label 5개; Malignant 15,723·unclassified 7,088세포는 대응 없음. 유전자 symbol 교집합 18,624개(HVG 후보는 양쪽에 유일한 symbol).

## A1 — N×D 진단 (`n_by_d_stress.tsv`, `n_by_d_effects.tsv`, `figures/fig1_n_by_d.png`; 인공 희소화)

**관측** (평가 donor 10명, 대상별 PCA는 대상 세포 없이 적합)

| 대상 | head | D 효과(1→4, N 평균) | N 효과(32→128, D 평균) | D−N |
|---|---|---|---|---|
| B | LR_class | +0.085 [0.045, 0.131] | +0.042 [0.029, 0.055] | +0.043 [0.012, 0.079] |
| B | P1_class | +0.117 [0.075, 0.165] | +0.043 [0.031, 0.055] | +0.074 [0.042, 0.112] |
| Endothelial | LR_class | +0.035 [0.016, 0.058] | +0.035 [0.017, 0.056] | +0.000 [−0.016, 0.020] |
| Endothelial | P1_class | +0.018 [0.004, 0.031] | +0.003 [−0.008, 0.017] | +0.014 [−0.004, 0.031] |
| Fibroblast | LR_class | +0.172 [0.106, 0.238] | +0.113 [0.077, 0.150] | +0.059 [0.014, 0.114] |
| Fibroblast | P1_class | +0.088 [0.062, 0.115] | +0.029 [0.001, 0.060] | +0.059 [0.024, 0.100] |

- 같은 32세포를 4명에서 받으면(Fibroblast LR 0.790) 1명에서 128세포를 받을 때(0.721)보다 높다. B도 같은 방향(0.809 vs 0.766).
- D가 늘면 recall과 오탐이 함께 늘었다(예: Fibroblast LR N=128: recall 0.34→0.72, 오탐/donor 21→46).

**추론:** 사전 규칙상 B·Fibroblast에서 "환자 지지도 가설 강화", Endothelial은 D 효과는 있으나 N보다 크다고 할 수 없음.

**미검증:** D 효과가 생물학적 환자 간 변동 때문인지, 시료·부위·배치 구성 차이 때문인지는 이 설계로 분리되지 않는다.
