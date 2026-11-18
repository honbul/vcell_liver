# R9 RESULTS — 관측 / 추론 / 미검증 분리

## 한눈에 (최종)

| 단계 | 결과 | 판정 |
|---|---|---|
| A0 | 배포 주석은 대분류 6개뿐, 자연 희귀 대분류 없음 | 주장 범위 축소(§4): Tier 1 인공 희소화 + Tier 2 군집 탐색 |
| A1 N×D | B·Fibroblast에서 D 효과 > N 효과(CI>0), Endothelial은 우열 없음 | 인공 희소화에서 환자 지지도 가설 강화 |
| A2 2×2 | A11−A10 CI가 6개 조건 모두 0 포함; 다중 프로토타입 arm < logistic/MLP | 귀무가설 기각 못함, 선정 버전 LR_donor |
| A3 | Tier 1 보류·순위 개선 없음; Tier 2 Myeloid AURC만 개선; pseudo-unknown은 A11>A10이나 < LR | 보류 기여 주장 없음 |
| A4 (GSE151530) | LR_donor 외부 nat 0.998, sparse 0.88–0.93; A11−A10 B·Fib +, Endo − | 개발 결론(채택 안 함) 유지 |
| §10 행 | "adaptive와 같은 K 고정 mixture가 비슷"(다중 패턴 이득도 없음) | 완료된 음성 결과 |
| 외부 모델 | ProtoCloud가 sparse_Fib에서 모든 같은-표현 head보다 높음; scPoli는 혼재 | R9 방법이 이겼다고 할 수 없음 |

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

## A2 — 같은 총 프로토타입 2×2 + 기준선 (`a2_model_table.tsv`, `a2_contrasts.tsv`, `BASELINES.tsv`, `prototype_support.tsv`, `figures/fig2_2x2_contrasts.png`, `figures/fig3_natural_rare_support.png`)

바깥 LOPO 10 × seed 3 = 조건당 30 fit. K = 3×클래스 수. 주 지표: nat — all-label AUPRC(자연 희귀 대분류 없음), sparse — 대상 label AUPRC(인공 희소화, 대상 64세포 / 4 donor), lin — fold별 다환자 자연 희귀 군집 AUPRC(Tier 2, 군집 번호 재현, 평가 donor 9명).

**관측 — 주 지표 평균 [95% donor CI]**

| 모델 | nat | sparse_B | sparse_Endo | sparse_Fib | lin_Myeloid | lin_B |
|---|---|---|---|---|---|---|
| A00 (class, 고정) | 0.945 | 0.773 | 0.934 | 0.787 | 0.696 | 0.693 |
| A10 (donor, 고정) | 0.941 | 0.757 | 0.925 | 0.770 | 0.697 | 0.682 |
| A01 (class, 적응) | 0.945 | 0.771 | 0.918 | 0.801 | 0.690 | 0.707 |
| A11 (donor, 적응) | 0.943 | 0.773 | 0.907 | 0.787 | 0.704 | 0.702 |
| TL_donor (훈련손실 분할) | 0.947 | 0.774 | 0.920 | 0.785 | 0.712 | 0.689 |
| RA_donor (무작위 배분) | 0.940 | 0.768 | 0.920 | 0.777 | 0.704 | 0.702 |
| GMM_K | 0.941 | 0.819 | 0.962 | 0.816 | 0.693 | 0.580 |
| P1_donor (단일) | 0.948 | 0.811 | 0.916 | 0.805 | 0.692 | 0.720 |
| LR_class | 0.953 | 0.840 | 0.938 | 0.853 | 0.725 | 0.775 |
| LR_donor (선정 버전) | 0.948 | 0.832 | 0.908 | 0.798 | 0.725 | 0.670 |
| MLP_donor | 0.953 | 0.897 | 0.951 | 0.865 | 0.734 | 0.719 |

핵심 대비 A11−A10 (donor paired):

| 조건 | 평균 [CI] | +/− donor 수 |
|---|---|---|
| nat | +0.0022 [−0.0001, +0.0045] | 8/2 |
| sparse_B | +0.016 [−0.010, +0.041] | 5/5 |
| sparse_Endothelial | −0.018 [−0.042, +0.004] | 5/5 |
| sparse_Fibroblast | +0.018 [−0.022, +0.052] | 7/3 |
| lin_Myeloid (Tier 2) | +0.007 [−0.020, +0.035] | 6/3 |
| lin_B (Tier 2) | +0.019 [−0.008, +0.048] | 6/3 |

- A10−A00(환자 균형 손실, 고정 배분): nat −0.004 [−0.007, −0.000], sparse_B −0.016 [−0.032, −0.002], sparse_Endo −0.009 [−0.012, −0.007], sparse_Fib −0.017 [−0.030, −0.006] → 이 head에서는 환자 균형 손실이 class 균형보다 약간 낮았다.
- A11−TL_donor, A11−RA_donor: 어떤 sparse·lin 조건에서도 CI가 0을 포함(nat에서 A11−RA +0.003 [0.001, 0.005], A11−TL −0.004 [−0.008, −0.001]).
- 같은 K 다중 프로토타입 arm은 sparse 조건에서 단일 프로토타입·logistic·MLP보다 낮다(sparse_B: A10 0.757 vs P1_donor 0.811, LR_donor 0.832, MLP_donor 0.897).
- 배분 근거: donor-supported 배분은 한 donor만 지지하는 프로토타입 비율을 낮췄다(lin_Myeloid 27% vs 고정 48%, 훈련손실 69%; sparse 1–6% vs 19–22%). lineage 과제에서는 탐욕 단계의 약 절반이 허용 분할이 없어 고정 fallback으로 채워졌다.
- 비용(1 fit 중앙 CPU초): LR 6–15, P1 40–70, 고정 다중 프로토타입 100–230, donor-supported 배분 4,300–8,400(A01에 배분 비용 포함, A11은 같은 배분 재사용), 훈련손실 배분 370–935.

**추론**
- 사전 규칙(C2)상 A11 선정 기준 0/3, A10 기준 1/3 → 선정 버전은 `LR_donor`.
- §10: "adaptive와 같은 K 고정 mixture가 비슷" 행. 다만 다중 대표 패턴 자체의 이득도 관측되지 않았으므로 "다중 대표 패턴의 효과"도 주장하지 않는다. 귀무가설을 기각하지 못한 **완료된 음성 결과**다.

### A2 외부 프로토타입 모델 (전체 pipeline 비교, `BASELINES.tsv` external_full_pipeline 블록, `a2_external_model_contrasts.tsv`)

seed 0만, 같은 source 세포·label·fold HVG 2000개 raw count. scPoli는 transductive(시험 donor RNA로 query surgery), ProtoCloud는 공식 기본값(GPU 시간 때문에 nat + sparse_Fibroblast만). HCC08 scPoli 4 task는 scanpy/SciPy 비호환 때문에 exact kNN으로 재실행(편차). 같은 head 비교가 아니다.

| 조건 | scPoli | ProtoCloud | A11 | LR_donor | MLP_donor |
|---|---|---|---|---|---|
| nat | 0.932 [0.909, 0.951] | 0.921 [0.895, 0.944] | 0.943 | 0.948 | 0.953 |
| sparse_B | 0.818 [0.752, 0.877] | 미실행 | 0.773 | 0.832 | 0.897 |
| sparse_Endo | 0.782 [0.672, 0.871] | 미실행 | 0.907 | 0.908 | 0.951 |
| sparse_Fib | 0.627 [0.456, 0.777] | **0.920 [0.882, 0.954]** | 0.787 | 0.798 | 0.865 |

- donor paired: A11−scPoli nat +0.011 [−0.006, +0.024], sparse_B −0.045 [−0.096, −0.008], sparse_Endo +0.125 [+0.034, +0.235], sparse_Fib +0.161 [+0.006, +0.325]. A11−ProtoCloud nat +0.022 [+0.016, +0.029], sparse_Fib **−0.132 [−0.222, −0.051]**; LR_donor−ProtoCloud sparse_Fib −0.122 [−0.182, −0.063]; MLP_donor−ProtoCloud sparse_Fib −0.054 [−0.087, −0.023].
- 관측: ProtoCloud는 인공 희소화 Fibroblast에서 같은 표현의 모든 head보다 높았다(같은 정보 계약; 다만 자체 encoder·희귀 type 증강·raw count 입력이라 아키텍처 귀속은 불가). scPoli는 sparse_B에서 A11보다 높고 Endo·Fib에서 낮았다.
- 따라서 R9 방법(A11)은 ProtoCloud를 sparse_Fibroblast에서 이기지 못했고, scPoli도 sparse_B에서 이기지 못했다. ProtoCloud는 sparse_B·sparse_Endo·Tier 2에서 실행하지 않았으므로 그 조건의 비교는 미확인이다.

**미검증(가설로만 남김)**
- 다중 프로토타입 arm이 단일 프로토타입보다 낮은 이유(64세포에 3개 성분 → 과적합, 300 step 최적화, 수축 0.3 등)는 측정하지 않았다.
- Tier 2에서 fallback 비율이 높은 것이 결과에 미친 영향은 분리하지 않았다.

## A4 — 별도 코호트 1회 평가 (GSE151530 HCC; `a4_model_table.tsv`, `a4_contrasts.tsv`, `a4_per_donor.tsv`, `a4_unmapped_cells.tsv`, `a4_prior_shift.tsv`, `figures/fig4_external.png`)

고정 버전 commit `a9c47d9`(선정 버전 LR_donor; 2×2 네 arm·P1_donor·GMM_K 동시 평가). GSE149614 10명 전체로 학습, seed 0–2(sparse는 seed마다 다른 4 donor). 외부 donor 25명; 평가 donor는 해당 label 양성 ≥10인 donor만(nat 25, sparse_B 15, sparse_Endo 16, sparse_Fib 12). 이 코호트는 이전에 대분류가 채점된 적이 있으므로 blind 확인이 아니다.

**관측 — 주 지표 [외부 donor CI]**

| 모델 | nat (5 label) | sparse_B | sparse_Endo | sparse_Fib |
|---|---|---|---|---|
| LR_donor (선정) | 0.998 [0.996, 0.999] | 0.926 [0.895, 0.954] | 0.925 [0.891, 0.954] | 0.880 [0.821, 0.934] |
| P1_donor | 0.993 | 0.941 | 0.809 | 0.829 |
| GMM_K | 0.996 | 0.933 | 0.981 | 0.836 |
| A00 | 0.995 | 0.912 | 0.908 | 0.827 |
| A10 | 0.992 | 0.913 | 0.864 | 0.807 |
| A01 | 0.996 | 0.930 | 0.858 | 0.868 |
| A11 | 0.992 | 0.928 | 0.848 | 0.842 |

| 대비 (외부 donor paired) | nat | sparse_B | sparse_Endo | sparse_Fib |
|---|---|---|---|---|
| A11−A10 | +0.000 [−0.002, +0.003] | **+0.015 [+0.002, +0.030]** | **−0.016 [−0.030, −0.003]** | **+0.035 [+0.016, +0.055]** |
| A01−A00 | +0.001 [−0.001, +0.004] | +0.018 [+0.002, +0.036] | −0.050 [−0.078, −0.025] | +0.041 [+0.019, +0.063] |
| A11−LR_donor | −0.005 [−0.009, −0.002] | +0.003 [−0.011, +0.017] | −0.078 [−0.132, −0.033] | −0.038 [−0.068, −0.007] |
| A11−GMM_K | −0.004 [−0.007, −0.001] | −0.005 [−0.020, +0.010] | −0.133 [−0.192, −0.082] | +0.006 [−0.029, +0.042] |

- A11−A10의 macro-F1(5 label): sparse_B −0.027 [−0.046, −0.012], Endo +0.028 [+0.007, +0.058], Fib +0.021 [+0.001, +0.041] — AUPRC와 부호가 엇갈린다.
- Prior 차이: 외부 Myeloid 9.2%(개발 22.2%), Fibroblast 2.1%(3.2%), B 3.7%(5.1%), Hepatocyte 정답 없음, Malignant 31%·unclassified 14%.
- 대응 없는 세포: Malignant cells는 모든 모델·조건에서 82–99%가 Hepatocyte로 예측되었고, 보류 비율은 모델·조건에 따라 30–99%. 이는 taxonomy 대응 기술(탐색)이며 정확도가 아니다.
- Source 보정 3단계 출력(nat, known 세포): LR_donor는 세부 출력 54%·보류 46%에서 출력 오류 0.000; A11은 세부 63%·상위 12%·보류 26%에서 0.002; GMM_K 세부 82%에서 0.001. Source에서 정한 임계값(오류 ≤2%)이 외부에서는 보수적으로 전이되었다(coverage 보장 주장 없음).

**추론**
- 선정 버전(LR_donor)은 외부에서도 nat·sparse_Endo·sparse_Fib에서 2×2 arm들보다 높거나 같았다.
- 적응 배분(A11/A01)이 고정 배분(A10/A00)보다 나은 target이 외부에서 2개(B, Fib), 나쁜 target이 1개(Endo)다. 개발에서는 세 target 모두 CI가 0을 포함했으므로, 이 외부 양성은 사전 선정 규칙을 통과하지 못한 버전의 외부 관측이며 채택 근거로 쓰지 않는다.
- §10 표에서 "A11이 A10·고정 mixture를 넘고 외부에서도 유지" 행은 성립하지 않는다(개발 미충족, GMM_K·LR_donor 미초과).

**미검증**
- 외부에서 B·Fib에만 적응 배분 이득이 나타난 이유(외부 prior, 대상 세포 상태 구성, donor 수 12–15명의 변동)는 측정하지 않았다.

## A3 — source 보정·3단계 출력·pseudo-unknown (`risk_coverage.tsv`, `a3_calibration_per_donor.tsv`, `a3_unknown_summary.tsv`, `a3_unknown_per_donor.tsv`, `a3_contrasts.tsv`)

모든 모델에 같은 보정: source donor 3 group OOF → 온도 1개, τ_fine(OOF 세부 오류 ≤2%), τ_parent, 그 외 보류. 바깥 모델은 A2 seed 0. Pseudo-unknown: nat에서 대분류 하나씩 제거(6개 × 10 donor), seed 0.

**관측**
- nat 3단계 출력(10 donor 평균): 출력 오류 0.018–0.021로 모든 모델이 source 목표(2%) 근처. 세부 출력 비율 LR_class 0.70, A00 0.65, GMM_K 0.63, LR_donor 0.56, A11 0.55, A10 0.55; 보류 0.22–0.36.
- 온도 보정은 nat에서 ECE를 낮추지 못했다(예: LR_donor 0.018→0.034, A11 0.019→0.067). 보정 전후 모두 보고한다.
- AURC A11−A10: nat −0.003 [−0.008, +0.000], sparse_B +0.002 [−0.003, +0.008], sparse_Endo −0.000 [−0.003, +0.002], sparse_Fib −0.007 [−0.022, +0.004], lin_Myeloid −0.018 [−0.033, −0.005].
- lin_Myeloid(Tier 2): 2% 오류 기준을 만족하는 세부 출력이 거의 없어(세부 0.1–0.6%) 대부분 상위(Myeloid) 출력 또는 보류였다.
- sparse_Fibroblast에서 LR_class·LR_donor는 source 임계값이 시험 donor에서 세부 출력 0.1–2%만 허용(보류 79–86%)했다 — 보정 전이 실패 사례.
- Pseudo-unknown AUROC(max prob, 6 label·10 donor 평균): LR_class 0.901, LR_donor 0.888, GMM_K 0.876, A00 0.864, A11 0.861, P1_donor 0.861, A10 0.842. log density 점수: GMM_K 0.878, 프로토타입 arm 0.77–0.79. A11−A10 +0.019 [+0.005, +0.034], A11−LR_donor −0.027 [−0.048, −0.004]. 같은 임계값에서 unknown 거부율은 0.83–0.91, known 거부율 0.27–0.44(max prob).

**추론**
- Tier 1에서는 분류 향상과 보류 향상 모두 A11에서 관측되지 않았다. Tier 2 Myeloid에서만 "분류는 같고 순위(AURC)가 좋아짐"이 관측되었다(탐색, 군집 번호 재현).
- 미관측 상태 보류에서 적응 배분은 고정 배분보다 낫지만 단순 logistic보다 나쁘다 → 보류 개선을 R9 방법의 기여로 주장하지 않는다.

**미검증:** pseudo-unknown은 대분류 제거 실험이며 실제 새로운 세포 상태의 발견 능력이 아니다.

## 자원 (`run_ledger.tsv`, `run_ledger_summary.tsv`, `resource_summary.txt`)

- 결과 실행 CPU 시간 합계 약 676 CPU-h(프로세스 CPU 시간; smoke 포함 683), task wall 합계 약 98 h. GPU: scPoli 40 task 6.8 GPU-h, ProtoCloud 20 task 13.5 GPU-h(+ smoke 0.9) → 약 21.2 GPU-h(A40/A30).
- `assets/` 디스크 14 GB(env 7.5 GB, cache 4.7 GB, runs 1.1 GB). 세포 단위 예측(.npz 362개)은 서버에만 두고 sha256을 `manifests/r9_rare_state/prediction_index.tsv`에 기록.
