# R9 CLAIMS_AND_LIMITATIONS

(결과 수치는 [RESULTS.md](RESULTS.md). 이 문서는 무엇을 주장할 수 있고 무엇을 주장할 수 없는지만 정리한다.)

## 자료가 지지하지 않아 하지 않는 주장

- **전문가 세부 아형(희귀 세부 상태) 인식:** GSE149614 배포 주석에 세부 아형 이름이 없다. Tier 2 결과는 저자 `res.3` 비지도 군집 번호 재현이며, 군집 자체가 시험 환자 세포를 포함해 정의되었다.
- **희귀 암 클론·기능·약물 반응:** 어떤 결과도 이를 뜻하지 않는다. 보류(abstention)는 새 아형 발견이 아니다.
- **Hepatocyte의 외부 일반화:** GSE151530에 대응 label이 없다. Malignant cells를 Hepatocyte 정답으로 쓰지 않았다.
- **봉인된 시험 결과:** HCC02·HCC09는 G1에서, GSE151530 HCC는 phase2-v2에서 이미 채점되었다. GSE149614 결과는 개발 증거, GSE151530 결과는 "R9 고정 버전의 첫 외부 적용"이다.
- **인과 분리:** N×D의 D 효과는 "여러 환자 자료에 노출된 효과"이며 기술 변동과 생물학적 변동을 인과적으로 분리하지 않는다.
- **방법 class 일반화:** 결론은 시험한 설정(PCA50, 대각 공분산 수축 0.3, K=3×클래스, 300 step, 이 allocation 규칙)에 한정한다.

## 섞지 않은 것

- 자연 희귀(Tier 2 군집), 인공 희소화(Tier 1 sparse), 환자 단독 label은 표·결론을 따로 둔다.
- 같은 embedding head 비교(BASELINES `same_PCA50_representation`)와 외부 전체 pipeline 비교(`external_full_pipeline`)는 따로 둔다. scPoli는 시험 donor RNA로 적응하는 transductive 조건이다.
- 선택용 점수(내부 donor-group OOF)는 보정에만 쓰고 일반화 결과로 보고하지 않는다.

## 자료가 지지하는 주장 (관측, 범위 한정)

1. **인공 희소화에서 지지 donor 수의 효과(A1):** 대상 세포 수를 고정할 때 지지 donor를 1→4명으로 늘리면 새 donor AUPRC가 B·Endothelial·Fibroblast 모두에서 올랐고, B·Fibroblast에서는 세포 수 1→4배 효과보다 컸다(GSE149614 개발 LOPO, donor 10명, PCA는 대상 세포 없이 적합).
2. **적응적 프로토타입 배분의 증분 가치는 확인되지 않음(A2):** 같은 총 K에서 A11−A10이 6개 개발 조건 모두에서 CI가 0을 포함했다. 다중 프로토타입 head 전체가 같은 표현의 logistic·MLP보다 낮았다. 귀무가설(균형 학습/고정 배분으로 충분)을 기각하지 못했다.
3. **배분 규칙의 의도된 성질:** donor-supported 배분은 한 donor만 지지하는 프로토타입 비율을 고정·훈련손실 배분보다 크게 줄였다. 이것이 예측 성능 향상으로 이어진다는 근거는 없다.
4. **외부(GSE151530 HCC) 1회 적용(A4):** 선정 버전 LR_donor는 5개 대응 대분류에서 donor-macro AUPRC 0.998, 인공 희소화 대상에서 0.88–0.93. 외부에서 A11>A10이 B·Fibroblast에서, A11<A10이 Endothelial에서 관측되었으나 개발 단계 선정 규칙을 통과하지 못한 버전이므로 채택 근거가 아니다.
5. **보류(A3):** Tier 1에서 A11은 보류·순위(AURC)·미관측 label 검출에서 단순 logistic보다 낫지 않았다. Tier 2 Myeloid 군집 과제에서만 A11의 AURC가 A10보다 낮았다(탐색).

## 외부 모델과의 관계

- ProtoCloud(공식 기본값, inductive)는 sparse_Fibroblast에서 모든 같은-표현 head보다 높았다. R9 방법은 이 조건에서 ProtoCloud를 이기지 못했다. ProtoCloud는 GPU 시간 때문에 nat·sparse_Fibroblast만 실행했으므로 다른 조건의 우열은 미확인이다.
- scPoli(transductive)는 sparse_B에서 A11보다 높고 Endo·Fib에서 낮았다. HCC08 4개 task는 비호환 회피용 exact kNN으로 실행한 편차가 있다.
  > **v2 정정(R9-C2, seed 0 matched)**: 위 두 줄은 3-seed 평균 대 외부 seed 0 비교였다. 같은 seed 0 표본으로 짝지으면 ProtoCloud는 sparse_Fib에서 A11(−0.143 [−0.238, −0.050])·LR_donor(−0.111 [−0.162, −0.056])보다 높지만 MLP_donor와는 우열 불확실(−0.048 [−0.099, +0.001]). A11−scPoli는 sparse_B −0.041 [−0.123, +0.044], sparse_Fib +0.150 [−0.003, +0.315]로 우열 불확실, sparse_Endo +0.144 [+0.048, +0.259]만 A11 우세. 근거 `a2_external_model_contrasts_matched_seed0.tsv`.
- 두 모델 모두 입력·encoder·증강이 달라 "아키텍처만의 우열"로 해석하지 않는다.

## 미실행 (사유)

- §10 후속 대조(covariance 수축·regularization을 맞춘 대조, 다른 고정 표현(scGPT/Geneformer)에서의 재현, K 인접값): A11 채택 조건이 성립하지 않아 이번 연구에서 실행하지 않음.
- ProtoCloud의 sparse_B·sparse_Endo·Tier 2 조건, 외부 모델의 A4 외부 적용: GPU 시간 제약(사전 규칙).
- 3 seed 외부 모델 반복: seed 0만.
