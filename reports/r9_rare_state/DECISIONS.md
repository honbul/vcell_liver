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
