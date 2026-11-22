# 원천 세포주 일치도로 만든 약물별 배율의 효과

**약물별 배율은 기존 단일 배율의 편차 축소를 앞섰고, 일치도의 독립 기여는 승격 기준에 미달했다.** 표적 웰과 위치가 다른 기존 비간계열 원천 8개와 편차 축소 벡터를 사용했다. 배율은 원천 평균 반응의 로그 RMS와 원천 세포주 간 평균 코사인으로 정했다. JHH-5 305군과 JHH-7 88군을 기존 10 µM·24 h B1 노출 웰에서 평가했다. JHH-7 적합은 JHH-5에서만 했다. [사전 명세](JHH_DONOR_AGREEMENT_GAIN_PROTOCOL.md) · [JHH-5 원표](data/jhh_donor_agreement_gain/JHH5/score/paired.tsv) · [JHH-7 원표](data/jhh_donor_agreement_gain/JHH7/score/paired.tsv)

| 세포주 | 약물별 배율 후보의 비교군 | 상대 RMSE 개선 % [99% 구간] | Pooled R² 개선 %p [99% 구간] |
|---|---|---:|---:|
| JHH-5 | 기존 고정 배율 | 1.114 [0.692, 1.566] | 2.134 [0.996, 3.293] |
| JHH-7 | 기존 고정 배율 | 1.685 [0.437, 3.090] | 3.859 [1.122, 6.507] |
| JHH-5 | 원천 평균 크기만으로 정한 배율 | 0.125 [0.024, 0.252] | 0.243 [−0.006, 0.606] |
| JHH-7 | 원천 평균 크기만으로 정한 배율 | 0.269 [−0.125, 0.779] | 0.404 [−0.363, 1.507] |
| JHH-5 | 약물 간 일치도 순열 | 0.267 [0.037, 0.565] | 0.595 [0.010, 1.440] |
| JHH-7 | 약물 간 일치도 순열 | 0.324 [−0.123, 0.912] | 0.492 [−0.366, 1.777] |

기존 후보 대비 두 지표는 두 세포주에서 세 fold 모두 개선됐다. 원천 평균 크기만 쓴 대조도 기존 후보보다 RMSE가 JHH-5 0.990% [0.526, 1.461], JHH-7 1.419% [−0.184, 3.088] 낮았다. 크기 대조를 넘는 일치도 후보의 0.5% RMSE 추가 이득은 JHH-5에서 99% 구간으로 배제됐다. JHH-7 구간은 0과 유용한 개선을 모두 포함했다. 두 비교와 순열을 함께 요구한 원 승격 기준은 미통과다. 독립 파일럿에는 기존 동결 후보를 사용한다. [JHH-5 판정](data/jhh_donor_agreement_gain/JHH5/gate.json) · [JHH-7 판정](data/jhh_donor_agreement_gain/JHH7/gate.json) · [JHH-5 계수](data/jhh_donor_agreement_gain/JHH5/score/gains.tsv) · [JHH-7 계수](data/jhh_donor_agreement_gain/JHH7/score/gains.tsv)

명세 `786d90c` 이후 첫 예측에서 크기만 쓴 대조의 해당 없는 계수를 JSON `NaN`으로 기록해 중단됐다. `875b574`에서 `null`로 고쳐 같은 명세를 재실행했다. 최종 예측·동결·채점 10개 작업 모두 종료 코드 0이었고 두 세포주의 기존 후보 점수를 재현했다. R18과 독립 파일럿의 발현은 읽지 않았다. [실행 원장](data/jhh_donor_agreement_gain/jobs.tsv) · [시도 이력](data/jhh_donor_agreement_gain/attempts.tsv)
