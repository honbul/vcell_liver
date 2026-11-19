# R9 DECISIONS — cycle별 관측 → 가설 → 변경 → 예상 판별 결과

각 cycle 항목의 "계획" 부분은 해당 실행을 **제출하기 전에** 커밋했다(커밋 SHA를 함께 적는다). "결과" 부분은 실행 후 덧붙이며, 앞서 적은 계획·판정 규칙은 고치지 않는다.

---

## C0 — A0 inventory (완료, 2026-11-18)

**관측 (A0 출력: `label_inventory.tsv`, `donor_class_counts.tsv`, `rare_definition_by_fold.tsv`, `cohort_overlap.tsv`)**

- GSE149614 배포 metadata 열은 `Cell, sample, res.3, site, patient, stage, virus, celltype` 뿐이다. 저자 주석은 6개 대분류(B, Endothelial, Fibroblast, Hepatocyte, Myeloid, T/NK)이며, 원문이 언급한 "세부 아형" 이름은 배포 파일에 **없다**.
- 대분류는 10명 모두에서 관측되고(모든 label이 ≥10세포 donor 10명), 학습 fold 기준 최소 비율은 Fibroblast 2.55%(HCC06 제외 fold)다. 즉 **0.5%·1%·2% 어느 기준에서도 자연 희귀 대분류 label이 없다.**
- `res.3`은 저자가 배포한 Seurat 군집 번호(53개)로, 모두 한 대분류에 100% 포함된다(broad purity 1.0). 이름이 없는 비지도 군집이며 **모든 환자 세포를 함께 군집한 결과**(시험 환자 세포도 군집 정의에 쓰임)다. 학습 fold 기준 1% 이하 군집이 다수 있고, 여러 환자가 지지하는 군집(예: Myeloid c46·c52, B c40)과 한 환자 지배 군집(예: B c50=HCC02 85%, Hepatocyte c47·c42)이 모두 있다.
- GSE151530 HCC 부분: 25명, 32 시료, 50,023세포. 저자 `Type`은 T cells, B cells, TAMs, TECs, CAFs, Malignant cells, unclassified 7개. Hepatocyte 대응 label은 없다. 이 HCC 부분은 `research/phase2-v2`의 E00 외부 평가(2026-10-26)에서 대분류 예측이 이미 채점되었다 → 이 프로젝트에서 대분류 분류의 blind cohort가 아니다.
- G1의 고정 시험 환자 HCC02·HCC09는 G1 비교와 phase2-v2에서 반복 채점되었다 → 봉인된 시험 환자가 아니다.

**결정 (§4 규칙 적용, 모델 실행 전)**

1. "희귀 세부 상태 복원" 주장을 **저자 전문가 주석으로는 할 수 없다.** 주 과제(Tier 1)는 두 코호트 공통 대분류(5개 대응 + GSE149614 Hepatocyte)로 좁힌다. Tier 1에는 자연 희귀 label이 없으므로 희귀 상태 질문은 **인공 희소화**(N×D, 희소 2×2)로만 답한다.
2. 자연 희귀 상태는 **Tier 2 탐색 과제**로만 다룬다: 대분류(부모)를 안다고 가정하고 그 안의 `res.3` 군집 번호를 맞히는 과제(Myeloid 14개, B 5개 군집). 결과는 "저자 군집 번호 재현"이며 전문가 세부 상태 인식이나 생물학적 아형 검증이 아니다. 외부 코호트로 옮기지 않는다(대응 label 없음).
3. Tier 2의 fold별 label 구분(학습 donor에서만 정의):
   - 자연 희귀 다환자 label: 학습 세포 비율 ≤1% **그리고** ≥10세포를 가진 학습 donor ≥3명 → 주 지표 대상.
   - 소수 환자 희귀 label: ≤1%이고 해당 donor 2명.
   - 환자 단독 label: ≥10세포 donor ≤1명 → 보류·탐색 항목. 일반화 결론에 섞지 않는다.
   - 0.5%·2% 민감도는 `rare_definition_by_fold.tsv`에 둔다.
4. 평가 가능한 donor-label: 시험 donor에 양성 ≥10세포. 양성이 0인 donor-label AUPRC는 정의불가(NA)이며 0으로 넣지 않는다.
5. 외부 대응표(GSE151530 → GSE149614): T cells→T/NK, B cells→B, TAMs→Myeloid, TECs→Endothelial, CAFs→Fibroblast. Malignant cells·unclassified는 **대응 없음**(Hepatocyte로 추정하지 않음). 채점 전 고정.
6. 표현: G1 PCA 조리법(CP10k+log1p, 학습 분산 상위 2000 HVG, 학습 scaler, PCA50, PC 표준화)을 fold별 학습 donor에만 적합. HVG 후보는 두 코호트에 모두 유일하게 있는 gene symbol(18,6xx개)로 제한(유전자 목록만 사용; 발현값 미사용). Tier 2는 해당 lineage 학습 세포로만 HVG/PCA 적합.

---

## C1 — A1 N×D 진단 (계획; 실행 전 커밋)

**관측:** Tier 1에는 자연 희귀 label이 없다. B·Endothelial·Fibroblast는 대부분의 donor에서 ≥128세포(희소화 풀 확보 가능).

**가설:** 같은 희소 세포 수 N에서 지지 donor 수 D가 많을수록 새 환자에서 해당 label 인식(AUPRC)이 좋아진다.

**변경(실험):** 대상 t ∈ {B, Endothelial, Fibroblast}; 바깥 LOPO 10 fold; 대상 t의 학습 세포를 N ∈ {32,64,128}, D ∈ {1,2,4}로 제한(풀 = t 세포 ≥128인 학습 donor; donor당 N/D개 균등; 5회 회전, D 중첩 donor 부분집합). 비대상 클래스 학습풀·시험 donor 고정. 표현은 t 세포를 전혀 쓰지 않은 PCA(`excl_t`) → RNA 정보와 label을 함께 줄인 실험. head: class-balanced logistic(`LR_class`), 단일 프로토타입(`P1_class`). 참고값: t 학습 세포 전체(`full_target_pool`).

**예상 판별 결과 (판정 규칙, 사전 고정):**
- D 효과 = 평균_N[AUPRC(N,D=4) − AUPRC(N,D=1)], N 효과 = 평균_D[AUPRC(128,D) − AUPRC(32,D)]; donor별 paired 차이, donor bootstrap 95% CI.
- D 효과 CI 하한 > 0 이고 D 효과 > N 효과(차이의 CI 하한 > 0)면 "환자 지지도 가설 강화".
- D 효과 CI가 0을 포함하고 N 효과 CI 하한 > 0이면 "표본 수 우선".
- 둘 다 CI가 0을 포함하면 "label 정의·표현 우선 점검".
- 판정은 target×head별로 따로 보고하고, 셋이 엇갈리면 엇갈린다고 적는다.
- 회전 반복은 기술 반복이며 독립 n이 아니다(donor 내 평균 후 donor 단위 통계).

### C1 결과 (실행 `a1_nxd_v1`, 30 task, 2026-11-18; `n_by_d_stress.tsv`, `n_by_d_effects.tsv`, `figures/fig1_n_by_d.png`)

- 관측: 평가 donor 10명 전부(모든 target에서 시험 donor 양성 ≥10). D 효과(D 1→4, N 평균)는 6개 target×head 조합 모두 양수이고 CI 하한 > 0 (B LR +0.085 [0.045, 0.131], P1 +0.117; Endothelial LR +0.035, P1 +0.018; Fibroblast LR +0.172 [0.106, 0.238], P1 +0.088).
- 사전 판정 규칙 적용: **B·Fibroblast**(두 head 모두) — D 효과 > N 효과, 차이 CI 하한 > 0 → "환자 지지도 가설 강화". **Endothelial** — LR: D 효과(+0.035)와 N 효과(+0.035)가 같고 차이 CI가 0 포함; P1: N 효과 CI가 0 포함, D−N 차이 CI 0 포함 → 규칙의 세 분기 중 어디에도 정확히 들지 않음: "D 효과는 있으나 N보다 크다고 할 수 없음".
- 예: Fibroblast LR에서 N=32·D=4(0.790)가 N=128·D=1(0.721)보다 높다. D가 늘면 recall과 오탐(false positive)이 함께 늘어난다(AUPRC는 임계값 무관 순위 지표).
- 해석 한계: 인공 희소화 결과이며 자연 희귀 상태 결론이 아니다. D=1 조건은 한 환자의 부위 구성·배치 특성을 함께 담으므로 "환자 수 노출 효과"이지 기술·생물 변동의 인과 분리가 아니다(§7).

---

## C2 — A2 같은 총 프로토타입 2×2 + §8 기준선 (계획; 실행 전 커밋)

**관측:** C1에서 지지 donor 수가 세포 수보다 중요할 수 있다는 결과(B·Fibroblast). Smoke(배관 점검용, 결과 아님)에서 고정 k-means 배분이 Hepatocyte 프로토타입 일부를 한 donor 전용 모드(support 1)에 쓰는 것을 확인.

**가설:** 같은 총 K에서, source donor를 하나씩 뺀 내부 검증 이득이 반복되고 두 자식 모두 ≥2 donor 지지를 받는 분할에만 용량을 주는 배분(A11)이 고정 배분(A10)보다 새 donor 인식을 개선한다.

**변경:** 헤드 13종(`scripts/r9/r9arms.py`): LR_cell/LR_class/LR_donor, MLP_donor, P1_class/P1_donor, GMM_K, A00/A10/A01/A11, TL_donor(훈련 손실 기반 분할), RA_donor(무작위 배분). K = 3×클래스 수(Tier 1: 18, Myeloid: 42, B: 15). 대각 공분산+수축 0.3, 클래스 내 혼합가중치 합 1, 균일 클래스 prior, 300 step Adam(모든 프로토타입 arm 동일). 손실은 가중 손실만(샘플링 보정 없음). A01과 A11은 같은 배분(같은 seed)을 공유하고 손실만 다르다.

**조건:** `nat`(Tier 1 자연 비율), `sparse_{B,Endothelial,Fibroblast}_N64_D4`(인공 희소화, C1에서 D 효과가 있던 지점), `lin_Myeloid`, `lin_B`(Tier 2 자연 희귀 군집, 탐색). 바깥 LOPO 10 × seed 3.

**주 지표:** nat — donor-macro all-label AUPRC(자연 희귀 대분류 없음); sparse — 대상 label AUPRC; lin — fold별 정의된 자연 희귀 다환자 군집 AUPRC. 보조: macro-F1, Brier, NLL, 오탐.

**예상 판별 결과 / 판정 규칙(사전 고정):**
- 핵심 A11−A10: donor paired 차이, donor bootstrap 95% CI. 조건별로 따로 보고하고 자연 희귀(Tier 2)와 인공 희소(Tier 1 sparse)를 섞지 않는다.
- A4에 보낼 "선정 버전" 규칙: A11−A10이 sparse 3개 target 중 ≥2개에서 CI 하한 > 0 **이고** nat macro-F1에서 A11−A10 평균 ≥ −0.005이면 A11을 선정. 아니면 A10(또는 A10이 LR_donor보다 sparse 3개 중 ≥2개에서 CI 하한 > 0로 낫지 않으면 LR_donor)을 선정 버전으로 둔다. 어느 경우든 A4에는 2×2 네 arm과 LR_donor·P1_donor·GMM_K를 함께 고정해 한 번에 평가한다.
- A11이 A10을 못 넘고 A10이 A00을 넘으면 §10 "주 기여는 환자 균형 학습" 행; A11≈TL_donor면 "환자 지지도의 귀속 불명"으로 적는다.

---

## C3 — A3 source calibration · 3단계 출력 · pseudo-unknown 보류 (계획; 실행 전 커밋)

**관측:** C2 실행 중(결과 미확인 상태에서 작성). 분류 성능 비교만으로는 보류 품질 차이를 알 수 없다.

**가설:** 다환자 지지 프로토타입(A11)은 donor 특이 모드에 용량을 덜 쓰므로, 같은 분류 성능에서도 새 donor의 신뢰도 순위(risk–coverage)와 미관측 상태 보류가 고정 배분(A10)보다 낫다.

**변경:** 모든 모델에 같은 source 보정 기회: source donor 3 group OOF → 온도 1개(NLL 격자), τ_fine(OOF 세부 오류 ≤2%), τ_parent(나머지에서 상위 유형 오류 ≤2%), 그 외 보류. 바깥 모델은 A2 seed-0 학습본 그대로. Pseudo-unknown: nat 조건에서 대분류 u 하나를 source에서 제거(6개 각각), 내부 OOF에서 다른 label u'를 추가로 숨겨 source pseudo-unknown을 만들고 Youden J로 임계값 결정; 점수는 max prob(모든 모델)과 log density(프로토타입·GMM).

**예상 판별 결과 / 규칙:** (i) A11−A10의 AURC(낮을수록 좋음)와 unknown AUROC를 donor paired CI로 비교. (ii) A2에서 분류 차이가 없는데 A3에서 AURC/보류가 CI 하한 기준으로 좋아지면 §10 "분류는 같지만 보류가 좋아짐" 행. (iii) 둘 다 차이 없으면 보류 개선 주장 없음. 외부 코호트에서는 calibration 전이 성능만 보고하며 coverage·정확도 보장을 주장하지 않는다.

### C2b — §8-6 외부 프로토타입 모델 (계획; 실행 전 커밋)

- scPoli(scArches 0.6.1, PyPI)와 ProtoCloud(Ding-Group GitHub, commit 90c2678)를 A2와 같은 source 세포·label·같은 fold HVG 2000개 raw count로 학습. 조건 nat + sparse 3개, seed 0, 바깥 LOPO 10.
- scPoli는 공식 query surgery(시험 donor RNA로 비지도 적응)를 거치므로 **transductive**; ProtoCloud는 공식 기본값(클래스당 6 프로토타입, 100 epoch, 내부 10% 무작위 검증, 희귀 type 증강)으로 inductive.
- 이들은 입력 표현·사전 처리·용량이 다르므로 "전체 pipeline 비교" 표에 따로 둔다(같은 embedding head 귀속 비교 아님). 실행이 실패하거나 계약 안에서 공정하게 돌릴 수 없으면 사유를 적고 "이겼다"고 세지 않는다.
- Smoke(배관용, 결과 아님): scPoli 1 fold 약 9.5분(A40, peak CUDA 0.06 GiB 할당). ProtoCloud 시간은 smoke 종료 후 기록. ProtoCloud가 task당 30분을 넘으면 GPU 2장 제한 때문에 조건을 nat + sparse_Fibroblast로 줄이고 그 사실을 여기에 적는다.
- scArches 0.6.1의 `classify()`가 SciPy 1.14에서 제거된 `csr.A`를 쓰므로 query 행렬을 dense로 넘겼다(방법 변경 아님).
- (C2b 관측 추가, 실행 전) ProtoCloud smoke: 1 fold 학습 2,651초(약 44분, A40, peak CUDA 0.54 GiB, CPU 단일 스레드 속도 제한). 사전 규칙에 따라 ProtoCloud 조건을 **nat + sparse_Fibroblast_N64_D4**로 줄인다(`configs/r9/a2_ext_protocloud.json`). scPoli는 4개 조건 그대로.

---

## C4 — A4 별도 코호트 한 번 평가 (계획; A2 결과를 보기 전 커밋)

- 고정 설정: `configs/r9/a4_freeze.json`. GSE149614 10명 전체로 학습, GSE151530 HCC 25명(32 시료)에 한 번 적용. 조건 nat + sparse 3개(대상 세포를 GSE149614 10명 중 4명에서 64개; seed별 donor 회전), seed 0–2. 모델: 2×2 네 arm + LR_donor, P1_donor, GMM_K(모두 같은 PCA50 표현).
- 대응표와 점수 대상은 A0에서 고정(5개 대응 label, Malignant/unclassified는 known 정답 아님). 주 지표: nat — 외부 donor별 5개 대응 label AUPRC 평균(양성 ≥10인 label만), sparse — 대상 label AUPRC. 외부 donor 단위 paired 차이와 donor bootstrap CI.
- 보정: GSE149614 OOF만 사용. GSE151530에서는 아무 것도 적합하지 않는다(온도·임계값·prior 포함).
- **선정 버전**은 C2 규칙을 A2 개발 결과에 기계적으로 적용해 결정하고, A4 제출 직전 이 절 아래에 기록한 뒤 freeze commit을 만든다. 외부 결과를 본 뒤 선정 버전·임계값을 바꾸지 않는다. 이후 바꾸면 새 버전 이름을 쓰고 이 자료를 blind test라고 부르지 않는다.
- 외부 모델(scPoli, ProtoCloud)은 GPU 시간 제약으로 A4에서 실행하지 않는다(개발 LOPO 비교에만 사용) — 미실행 항목으로 보고.
- 이 코호트는 phase2-v2에서 대분류가 이미 채점되었으므로 결과는 "R9 고정 버전의 첫 외부 적용"이며 blind 확인이 아니다.

### C2 결과 — Tier 1 부분 (a2_main_v1, nat + sparse 3개 × 10 donor × 3 seed 완료; Tier 2는 실행 중)

- 관측(donor paired, 95% donor bootstrap CI, 주 지표):
  - 핵심 **A11−A10**: nat +0.0022 [−0.0001, 0.0045]; sparse_B +0.016 [−0.010, 0.041]; sparse_Endothelial −0.018 [−0.042, 0.004]; sparse_Fibroblast +0.018 [−0.022, 0.052]. CI 하한 > 0인 sparse target **0/3**.
  - A10−LR_donor: nat −0.007 [−0.016, −0.001]; sparse_B −0.075 [−0.123, −0.032]; sparse_Endothelial +0.016 [+0.000, +0.034]; sparse_Fibroblast −0.028 [−0.083, +0.021] → CI 하한 > 0 은 1/3.
  - 같은 K의 다중 프로토타입 arm(A00–A11, TL, RA)은 sparse_B에서 단일 프로토타입·logistic·MLP보다 낮다(예: A10 0.757, P1_donor 0.811, LR_donor 0.832, MLP_donor 0.897).
- **C4 선정 버전(사전 규칙 기계 적용): `LR_donor`** (PCA50 + L_donor,class logistic). A11 기준(≥2/3) 불충족, A10 기준(≥2/3) 불충족.
- A4는 freeze 설정대로 2×2 네 arm과 LR_donor·P1_donor·GMM_K를 모두 한 번에 평가하고, 선정 버전은 LR_donor로 표시한다.

### C2 결과 — 전체 (a2_main_v1 180 task 완료; `a2_model_table.tsv`, `a2_contrasts.tsv`, `prototype_support.tsv`, `figures/fig2_2x2_contrasts.png`, `figures/fig3_natural_rare_support.png`)

- Tier 2(자연 희귀 다환자 군집, 탐색) A11−A10: lin_Myeloid +0.007 [−0.020, +0.035], lin_B +0.019 [−0.008, +0.048] (평가 donor 각 9명). 두 lineage 모두 LR_class가 가장 높다(Myeloid 0.725, B 0.775).
- 배분 근거(`prototype_support.tsv`): donor-supported 배분은 단일 donor 지지 프로토타입 비율을 줄였다(lin_Myeloid 27% vs 고정 48%, 훈련 손실 기반 69%; sparse 1–6% vs 19–22%). 그러나 lineage 과제에서 탐욕 단계의 50%(Myeloid 420/840), 47%(B 141/300)가 허용 가능한 분할이 없어 고정 fallback으로 채워졌다.
- §10 판정: A11은 A10을 어떤 조건에서도 CI 기준으로 넘지 못했고, A10도 A00을 넘지 못했다(nat·sparse에서 A10−A00 음수, CI 하한<0 또는 상한<0). 같은 K의 다중 프로토타입 arm 전체가 sparse_B에서 단일 프로토타입·logistic보다 낮다. → "adaptive와 같은 K 고정 mixture가 비슷" 행에 해당하나, 다중 대표 패턴 자체의 이득도 관측되지 않았다. **귀무가설(환자 균형 샘플링이나 고정 다중 프로토타입이면 충분, 적응적 배분 추가 가치 없음)을 기각하지 못함.**
- 다음 결정: §10 행의 후속 실험(covariance 수축·regularization을 맞춘 대조)은 이 cycle에서 실행하지 않는다(미실행 항목으로 보고). A3·A4는 계획대로 진행.
- (C2b 실행 중 관측) scPoli task 7(nat, HCC08)이 scanpy 1.10.4 `neighbors()`의 `csr_array.getnnz`(SciPy 1.14에서 제거) 호출로 실패. 수치에 영향 없는 alias 복원을 `a2_ext.py`에 추가하고, 실패한 task만 같은 run ID에 다시 실행한다(기존 결과 덮어쓰기 없음).

### C4 결과 (a4_external_v1, 12 task 완료, freeze a9c47d9 이후 1회 채점; `a4_model_table.tsv`, `a4_contrasts.tsv`, `a4_per_donor.tsv`, `a4_unmapped_cells.tsv`, `a4_prior_shift.tsv`, `figures/fig4_external.png`)

- 선정 버전 **LR_donor**: nat 0.998 [0.996, 0.999] (25명), sparse_B 0.926 [0.895, 0.954] (15명), sparse_Endothelial 0.925 [0.891, 0.954] (16명), sparse_Fibroblast 0.880 [0.821, 0.934] (12명).
- 핵심 A11−A10(외부 donor paired): nat +0.0002 [−0.0018, +0.0029]; sparse_B **+0.015 [+0.002, +0.030]**; sparse_Endothelial **−0.016 [−0.030, −0.003]**; sparse_Fibroblast **+0.035 [+0.016, +0.055]**. 개발(A2)에서는 세 sparse 모두 CI가 0을 포함했고 부호는 같았다(B +, Endo −, Fib +).
- A11−LR_donor: nat −0.005 [−0.009, −0.002], sparse_B +0.003 [−0.011, +0.017], sparse_Endo −0.078 [−0.132, −0.033], sparse_Fib −0.038 [−0.068, −0.007]. A01−A00(class 손실에서 적응 vs 고정)도 B +0.018 [0.002, 0.036], Fib +0.041 [0.019, 0.063], Endo −0.050.
- 판정: 외부에서 적응 배분이 고정 배분보다 나은 target이 2개(B, Fib), 나쁜 target이 1개(Endo). 선정 버전(LR_donor)이나 같은 K 고정 GMM보다 일관되게 낫지는 않다. 선정 버전과 사전 규칙은 바꾸지 않는다. 외부 결과를 근거로 한 새 버전은 만들지 않았다.
- (C2b 추가) 원인 재확인: r9ext의 SciPy가 1.15.3으로 설치되어 scanpy 1.10.4의 근사 kNN(pynndescent) 경로가 `csr_array`를 반환하고 anndata 0.10.9가 이를 거부한다(약 9천 세포 이상 query donor = HCC08에서만 발생; task 7·17 실패, 27·37도 같은 donor). alias만으로는 해결되지 않아, **HCC08 4개 task만** `R9_SCPOLI_EXACT_KNN=1`로 scPoli 프로토타입 초기화 내부 kNN을 scanpy exact(sklearn) kNN으로 바꿔 재실행한다. 기본 경로와 다른 편차이므로 결과 표에 표시한다.

### 운영 변경 (OPS-1, 2026-11-18 22:48 KST; 과학적 설정 변경 아님)

- 조정 에이전트가 전달한 사용자 승인(다른 연구 종료, GPU 약 24/28 여유)에 따라 R9 GPU 동시 사용 한도를 2 → **최대 8**로 올렸다(여유 GPU의 1/3 이상은 남김).
- 변경: ProtoCloud array `26382` ArrayTaskThrottle 1→4, scPoli array `26374` 1→3, scPoli HCC08 exact-kNN 재실행 `26574` throttle 1(의존성 afterany:26374 유지). 합계 ≤8.
- 실행 중 task는 취소하지 않았고, 코드·설정·seed·조건은 그대로다. 실행 시점만 바뀐다.

### C3 결과 (a3_calib_v1 50 task, a3_unknown_v1 60 task; `risk_coverage.tsv`, `a3_calibration_per_donor.tsv`, `a3_unknown_summary.tsv`, `a3_unknown_per_donor.tsv`, `a3_contrasts.tsv`)

- AURC(낮을수록 좋음) A11−A10: nat −0.003 [−0.008, +0.000], sparse_B +0.002 [−0.003, +0.008], sparse_Endo −0.000 [−0.003, +0.002], sparse_Fib −0.007 [−0.022, +0.004], **lin_Myeloid −0.018 [−0.033, −0.005]**.
- Pseudo-unknown(대분류 하나 제거, max prob): A11−A10 AUROC +0.019 [+0.005, +0.034], 그러나 A11−LR_donor −0.027 [−0.048, −0.004]; 최고는 LR_class 0.901, LR_donor 0.888, GMM_K(log density) 0.878. log density 점수에서 A11−A10 AUROC −0.003 [−0.014, +0.010].
- 판정: Tier 1에서는 보류·순위 개선이 없다(규칙 (iii)). Tier 2 lin_Myeloid에서만 같은 분류 성능에서 AURC가 좋아졌으므로 §10 "분류는 같지만 보류가 좋아짐" 행이 **Tier 2 탐색 과제 한정**으로 관측되었다. 미관측 상태 보류에서 A11은 A10보다 낫지만 단순 logistic보다 나쁘다.
- OPS-2 (2026-11-19 00:1x KST, 운영 변경만): scPoli 본 array 종료 후 HCC08 재실행 `26574` throttle 1→3, ProtoCloud `26382` throttle 4→7. R9 GPU 동시 사용 합계는 계속 ≤8.

### C2b 결과 (a2_ext_v1_scpoli 40 task, a2_ext_v1_protocloud 20 task; `BASELINES.tsv`, `a2_external_model_contrasts.tsv`)

- scPoli: nat 0.932, sparse_B 0.818, sparse_Endo 0.782, sparse_Fib 0.627. ProtoCloud: nat 0.921, sparse_Fib 0.920.
- A11−ProtoCloud sparse_Fib −0.132 [−0.222, −0.051]; A11−scPoli sparse_B −0.045 [−0.096, −0.008]. → R9 방법이 두 외부 모델을 일관되게 이겼다고 할 수 없다. ProtoCloud 미실행 조건은 미확인.
  > **v2 정정(R9-C2)**: 위 수치는 3-seed 평균 대 외부 seed 0. seed 0 matched: A11−ProtoCloud sparse_Fib −0.143 [−0.238, −0.050], A11−scPoli sparse_B −0.041 [−0.123, +0.044](우열 불확실). `a2_external_model_contrasts_matched_seed0.tsv`.

---

## 종료 (2026-11-19)

- 최종 판정: §10 "adaptive와 같은 K의 고정 mixture가 비슷" 행(다중 패턴 이득도 없음). 적응적 배분의 증분 가치는 주장하지 않는다. 주 기여도 "환자 균형 학습"으로 볼 수 없다(A10−A00 음수).
- 다음 가설(실행하지 않음, 열린 결정): (1) 다중 프로토타입 head가 단일 프로토타입보다 낮은 원인 분리(수축·regularization·최적화 예산 대조), (2) A1의 D 효과를 부위·배치를 맞춘 donor 부분집합으로 재검증, (3) ProtoCloud가 희소 Fibroblast에서 강한 이유(희귀 type 증강 vs encoder)를 같은 표현에서 분리. 실행하려면 새 버전 이름과 새 cycle로 시작한다.
