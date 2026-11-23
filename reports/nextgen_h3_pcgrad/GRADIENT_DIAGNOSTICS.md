# 배치별 기울기 진단

A40에서 18개 inner fold × 8개 arm × 384 update를 추적했다. 목적 쌍 165,888건 중 두 norm이 1e−8을 넘은 경우는 165,888건이다. G1의 음의 cosine 비율은 1.15%다. 아래 수치는 투영 전 공유 인코더의 기울기이며, 가중 norm 비율에는 목적 가중치를 곱했다.

| Arm | 목적 A | 목적 B | 중앙 cosine | 중앙 norm A/B | 가중 A/B | 충돌 | 유효 update |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G0 | drug_pearson | cell_listnet | +0.1409 | 0.29 | 0.67 | 0.71% | 6,912 |
| G0 | smooth_l1 | cell_listnet | +0.2914 | 0.14 | 3.96 | 2.31% | 6,912 |
| G0 | smooth_l1 | drug_pearson | +0.4835 | 0.48 | 5.98 | 0.19% | 6,912 |
| G1_w0 | drug_pearson | cell_listnet | +0.1570 | 0.26 | 0.59 | 0.35% | 6,912 |
| G1_w0 | smooth_l1 | cell_listnet | +0.2779 | 0.15 | 4.21 | 2.81% | 6,912 |
| G1_w0 | smooth_l1 | drug_pearson | +0.5252 | 0.57 | 7.09 | 0.13% | 6,912 |
| G1_w1 | drug_pearson | cell_listnet | +0.1644 | 0.24 | 1.11 | 0.25% | 6,912 |
| G1_w1 | smooth_l1 | cell_listnet | +0.2817 | 0.15 | 4.19 | 2.46% | 6,912 |
| G1_w1 | smooth_l1 | drug_pearson | +0.5390 | 0.60 | 3.78 | 0.09% | 6,912 |
| G1_w2 | drug_pearson | cell_listnet | +0.1534 | 0.27 | 0.31 | 0.35% | 6,912 |
| G1_w2 | smooth_l1 | cell_listnet | +0.2604 | 0.15 | 2.19 | 3.75% | 6,912 |
| G1_w2 | smooth_l1 | drug_pearson | +0.5196 | 0.57 | 7.17 | 0.16% | 6,912 |
| G2_w0 | drug_pearson | cell_listnet | +0.1569 | 0.26 | 0.59 | 0.38% | 6,912 |
| G2_w0 | smooth_l1 | cell_listnet | +0.2776 | 0.15 | 4.22 | 2.81% | 6,912 |
| G2_w0 | smooth_l1 | drug_pearson | +0.5247 | 0.57 | 7.10 | 0.16% | 6,912 |
| G2_w1 | drug_pearson | cell_listnet | +0.1645 | 0.24 | 1.11 | 0.26% | 6,912 |
| G2_w1 | smooth_l1 | cell_listnet | +0.2814 | 0.15 | 4.20 | 2.50% | 6,912 |
| G2_w1 | smooth_l1 | drug_pearson | +0.5387 | 0.60 | 3.78 | 0.09% | 6,912 |
| G2_w2 | drug_pearson | cell_listnet | +0.1531 | 0.27 | 0.31 | 0.32% | 6,912 |
| G2_w2 | smooth_l1 | cell_listnet | +0.2606 | 0.15 | 2.19 | 3.86% | 6,912 |
| G2_w2 | smooth_l1 | drug_pearson | +0.5194 | 0.57 | 7.18 | 0.17% | 6,912 |
| G3 | drug_pearson | cell_listnet | +0.1473 | 0.29 | 0.67 | 0.94% | 6,912 |
| G3 | smooth_l1 | cell_listnet | +0.2417 | 0.15 | 4.26 | 5.82% | 6,912 |
| G3 | smooth_l1 | drug_pearson | +0.5227 | 0.52 | 6.52 | 1.06% | 6,912 |

## 배치 지지

밀도는 관측 쌍을 방문 세포×약물 슬롯으로 나눈 값이다. Pearson은 약물당 유효 관측 8개와 비상수 실측, ListNet은 세포당 관측 2개를 요구한다.

| 구성 | 평균 관측 | 밀도 | Pearson 그룹 | ListNet 그룹 | 방문 세포 중 ListNet |
| --- | --- | --- | --- | --- | --- |
| B_pcgrad_block4x3 | 1066.0 | 72.47% | 19.1 | 75.9 | 99.20% |
| B_pcgrad_block6x2 | 1066.0 | 72.47% | 28.6 | 51.0 | 99.77% |
| B_sum_block4x3 | 1066.0 | 72.47% | 19.1 | 75.9 | 99.20% |
| B_sum_block6x2 | 1066.0 | 72.47% | 28.6 | 51.0 | 99.77% |
| G1_w0 | 1066.0 | 6.51% | 56.2 | 245.5 | 86.20% |

## 실행 시간

감사 실행은 모든 arm에서 목적별 기울기를 계산한다. GPU-hour는 A40 적합의 동기화된 실측 wall-time이다. 감사 없는 재실행은 한 fold에서 같은 update 수로 측정했다.

| 감사 arm | 중앙 초 / 적합 | 총 GPU-hour |
| --- | --- | --- |
| G0 | 12.55 | 0.0593 |
| G1_w0 | 6.34 | 0.0395 |
| G1_w1 | 6.79 | 0.0384 |
| G1_w2 | 6.83 | 0.0394 |
| G2_w0 | 8.43 | 0.0418 |
| G2_w1 | 7.48 | 0.0410 |
| G2_w2 | 6.22 | 0.0358 |
| G3 | 6.86 | 0.0390 |

| 감사 없는 arm | 초 / 적합 | GPU-hour | 감사 예측과 최대 차이 |
| --- | --- | --- | --- |
| G0 | 6.898 | 0.00192 | 0 |
| G3 | 3.462 | 0.00096 | 2e-07 |
| G1_w0 | 5.269 | 0.00146 | 1.06e-05 |
| G2_w0 | 5.491 | 0.00153 | 0 |

G1 직접 역전파와 목적별 합의 수치 차이를 별도로 기록했다. 최종 H3a 적합은 G1과 G2에 동일한 목적별 역전파를 썼다. [에포크 요약](H3_GRADIENT_BY_EPOCH.tsv) · [실행 원장](H3_INNER_RUNTIME.tsv)

원시 배치별 epoch/update, cosine, 양쪽 norm, 두 norm 비율, 충돌 및 CUDA 이벤트 시간은 G1 H3 루트의 round1/<split>/gradients-inner*.parquet와 round2-block/<split>/gradients-inner*.parquet에 있다. 배치 파일 해시는 inner 분석 영수증에 기록했다. [Inner 분석 영수증](inner_analysis_receipts.json) · [동결과 결과](ROUND1_TO_3_INNER_TUNING.md)
