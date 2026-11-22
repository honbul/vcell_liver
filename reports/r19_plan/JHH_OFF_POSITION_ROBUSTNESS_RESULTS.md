# 위치 분리 성인 HCC 결과의 화합물 기여와 군집 민감도

**공개 VC 최선 대비 개선 방향은 scaffold 단위로 재표집해도 유지됐다.** 단일 배율보다 RMSE가 낮은 화합물은 JHH-5에서 260/305군(85.2%), JHH-7에서 64/88군(72.7%)이었다. 개선 폭은 영향이 큰 화합물에 집중됐다. 특히 JHH-7 상위 20군이 단일 배율 대비 양의 SSE 절감의 89.4%를 차지했다. 이 분석은 두 세포주의 기존 B1 노출 동결 예측을 재채점한 것이다. [명세](JHH_OFF_POSITION_ROBUSTNESS_PROTOCOL.md) · [JHH-5 기여 분포](data/jhh_off_position_robustness/JHH5/distribution.tsv) · [JHH-7 기여 분포](data/jhh_off_position_robustness/JHH7/distribution.tsv)

| 평가 | Parent / scaffold / plate | 상대 RMSE 개선 % [99% 구간] | Pooled R² 개선 %p [99% 구간] |
|---|---:|---:|---:|
| JHH-5, scaffold 재표집 | 305 / 276 / 6 | 0.751 [0.555, 0.981] | 1.662 [1.146, 2.233] |
| JHH-5, scaffold와 plate 교차 가중 | 305 / 276 / 6 | 0.751 [0.571, 0.989] | 1.662 [0.722, 2.827] |
| JHH-7, scaffold 재표집 | 88 / 88 / 1 | 1.278 [0.675, 1.968] | 2.945 [1.569, 4.307] |

표는 매 재표집에서 지표별 공개 VC 최선을 다시 선택한 결과다. JHH-7은 화합물당 scaffold가 하나씩이고 평가 plate가 한 장이라 plate 간 재현성은 추정할 수 없다. JHH-5에서 동일 교차 가중을 B0에 적용한 R² 개선은 7.612%p [−0.855, 16.537]였고, 화합물별 RMSE는 111/305군에서 낮았다. [JHH-5 군집 원표](data/jhh_off_position_robustness/JHH5/clustered.tsv) · [JHH-7 군집 원표](data/jhh_off_position_robustness/JHH7/clustered.tsv)

단일 배율 대비 양의 SSE 절감 상위 20군을 빼면, JHH-5의 최선 VC 대비 개선은 RMSE **0.461%**, R² **0.834%p**로 줄었다. JHH-7은 **0.218%**, **0.459%p**였다. 원 기준의 0.5% RMSE 개선을 두 코호트 모두 넘지 못했다. RMSE 개선을 기준으로 상위 20군을 빼도 같은 결론이었다. 상위군 제거는 관측 정답을 사용한 민감도 분석이다. 이 제거 결과로 실제 추론 집단을 정의하거나 기존 기술 판정을 바꾸지 않았다. [JHH-5 상위군 제외](data/jhh_off_position_robustness/JHH5/top_removed.tsv) · [JHH-7 상위군 제외](data/jhh_off_position_robustness/JHH7/top_removed.tsv)

명세 `350fdd5`를 커밋한 뒤 CPU 작업 `31822_0`과 `31822_1`에서 실행했다. 두 작업은 종료 코드 0이었고 프로세스 시간은 각각 9.55초와 5.22초였다. 원 공개 VC 점수와 99% 구간을 재현한 뒤 기여 계산을 수행했다. 화합물별 기여 1,572행은 G1 실행 경로에 보관했다. 학습과 예측 재생성 및 R18 예약 자료 접근은 없었다. [실행 원장](data/jhh_off_position_robustness/launch.tsv) · [JHH-5 영수증](data/jhh_off_position_robustness/JHH5/receipt.json) · [JHH-7 영수증](data/jhh_off_position_robustness/JHH7/receipt.json)
