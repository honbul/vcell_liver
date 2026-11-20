# R9 eligible_tasks — 실제로 평가 가능한 과제 (A0 기준, 모델 실행 전 확정)

근거 표: `label_inventory.tsv`, `donor_class_counts.tsv`, `rare_definition_by_fold.tsv`. 희귀 정의는 fold마다 **학습 donor에서만** 계산(비율 ≤1%, 민감도 0.5%·2%). 평가 가능한 donor-label = 시험 donor 양성 ≥10세포.

## 1. 알려진 희귀 상태의 환자 간 일반화 (자연 희귀)

| 계층 | 가능 여부 | 근거 |
|---|---|---|
| Tier 1 대분류 | **불가** | 학습 fold 최소 비율 Fibroblast 2.55%. 0.5/1/2% 어느 기준에도 희귀 대분류 없음 |
| Tier 2 `res.3` 군집 (탐색) | 가능, 단 "군집 번호 재현" | Myeloid: c41(3–4 donor), c46(7–8), c52(5–6)는 모든 fold에서 ≤1% 이고 ≥3 donor 지지 → 다환자 자연 희귀. c44는 fold에 따라 2–3 donor. B: c40(7–8 donor) |

Tier 2 평가 가능한 시험 donor 수(양성 ≥10): c46 8명, c52 6명, c41 4명, c44 3명, c40 8명. 소수 donor라 donor별 값을 반드시 함께 본다.

## 2. 인공 희소화 (기전 진단)

| 대상 | 풀(≥128세포 donor) | 사용 |
|---|---|---|
| B | 9명(HCC01 제외: 11세포) | A1 N×D, A2 `sparse_B_N64_D4` |
| Endothelial | 9명(HCC02 제외: 14세포) | 동일 |
| Fibroblast | 7명(HCC01·04·05·06·08·09·10; HCC07은 118세포로 제외) | 동일 |

학습 세포 수 N과 지지 donor 수 D를 분리 통제한다. 표현은 대상 세포 없이 적합. 결과는 "표본을 줄인 과제"에 한정한다.

## 3. 환자 단독 상태와 미관측 상태 보류

- 환자 단독 군집(≥10세포 학습 donor ≤1명): B c50(HCC02 85%), Myeloid c53(1–2 donor), T/NK c51, Hepatocyte c47·c42 등. 알려진 label 일반화 결론에 넣지 않고 탐색 표에만 둔다. c50은 HCC02가 시험일 때 학습에 없는 label이 된다.
- 미관측 상태 보류: Tier 1에서 대분류 하나씩 학습에서 제거하는 leave-label-out(pseudo-unknown) 과제로 평가(A3). 실제 "새 상태 발견"이 아니다.

## 4. 별도 코호트 (A4)

- GSE151530 HCC 25명 / 32 시료 / 50,023세포. 대응 label 5개(B, T/NK, Myeloid, Endothelial, Fibroblast). Hepatocyte·세부 군집은 대응 없음 → 외부에서 평가하지 않는다.
- Malignant cells(15,723)·unclassified(7,088)는 known 정답으로 채점하지 않고 예측 분포·보류율만 기술한다.
- 외부 prevalence: B 3.7%, CAFs 2.1%, TECs 4.7% (HCC 세포 전체 기준). 개발 코호트와 prior가 다르다.

## 제외한 주장

- 전문가 세부 아형(원문 "13개 세부 아형") 인식: 배포 주석 없음 → 주장하지 않음.
- 희귀 암 클론, 기능, 약물 반응: 자료가 지지하지 않음.
