# 연구 R8 — 자료 노출 기록 (EXPOSURE)

**작성일:** 2026-11-17 · 학습 모델을 고르기 **전**, HCC 평가 **전**에 작성했다.

## 1. 한 줄 요약

이 연구의 HCC 평가는 새로 숨겨 둔 시험이 아니다. **이미 여러 번 채점된 공개 벤치마크에서, 미리 적어 둔 계획대로 하는 후속 평가(pre-specified follow-up on an already-seen public benchmark)** 다.

## 2. 무엇이 이미 노출됐나

| 항목 | 상태 |
|---|---|
| HCC 세 계열(HUH7, JHH5, JHH7)의 Level 3 자료 | 연구 B에서 처리·채점했고, 연구 R2가 같은 분할로 K=8/16/32 점수를 한 번 더 봤다 |
| 시험 분할(약물 그룹 단위, split seed 1–3) | 연구 B·R2의 것을 **그대로** 쓴다 |
| 이 연구의 설계 동기 | R2의 HCC 결과("학습된 잔차 보정이 닫힌 식 기준선보다 조금 낮았지만 평균 보정과 구분이 덜 됐다")와 R5의 음성 결과에서 나왔다. 즉 **질문 자체가 HCC 시험 결과의 영향을 받았다** |
| R2의 K=16 HCC 숫자 | 과거 기록으로만 둔다. 이 연구의 새 점수로 옮겨 적지 않는다. XC pool 제외·family 균형 f0로 **모든 기준선을 다시 계산**한다 |

## 3. 이 연구에서 지키는 것

- 동결 전에는 HCC 발현 값을 읽지 않는다. store 작성은 HCC의 **메타데이터(plate·well 수)** 만 읽었다(`store_manifest.json`의 `expression_values_read: false`). HCC 값을 여는 함수는 동결 manifest가 Git에 있고 최종 평가 작업일 때만 동작한다.
- f0, feature 축, 관측 공분산 상수(v_r, v_e, v_s), 축소 상수 κ, 학습률·step 수, TranSiGen 미세조정 설정은 모두 XC pool을 뺀 **132개 source 세포주(126 family)** 만으로 정한다.
- 모델 후보는 source-train family 안의 3-fold family 교차검증으로만 고른다. R2가 검증에 쓰던 family(11개 세포주)는 **고정된 확인**에만 쓴다.
- 기준 family 목록(`reference_families.tsv`)은 store를 만든 시점에 고정했다.
- target에서 쓰는 처리 후 라벨은 K개 support가 전부다. target 검증 라벨은 0개다. query의 처리 후 RNA는 입력·정규화·신뢰도 계산 어디에도 쓰지 않는다.

## 4. 그래도 남는 한계

- 같은 벤치마크를 세 번째로 쓰는 것이므로 HCC 점수는 **독립적인 확인이 아니다.**
- 확인 family의 닫힌 식 점수(C0/C1)는 진단 단계에서 이미 한 번 계산됐다(`closed_form_summary.tsv`). 닫힌 식 방법의 상수는 dev episode로만 골랐지만, 확인 family 점수를 본 뒤에 학습 모델 실험이 진행된다는 점은 적어 둔다.
- 독립적인 확인에는 이 계열 연구에서 한 번도 쓰지 않은 세포주가 필요하다. 이번에는 하지 않는다.
