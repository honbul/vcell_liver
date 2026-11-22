# 동일 약물의 JHH-5·JHH-7 반응 차이

**현재 동결 후보는 두 세포주의 약물별 반응 차이에서 차이 0 기준선을 앞서지 못했다.** CMap 10 µM·24 h의 공통 86군을 JHH-5 ASG002와 JHH-7 단일 plate에서 짝지었다. 두 판의 RNA 웰 위치 인덱스는 86군에서 모두 같았다. 그중 84군은 두 예측이 참조한 비간계열 원천의 처리 웰과 역할별 DMSO가 정확히 같았다. [채점 전 명세](JHH_OFF_POSITION_INTERACTION_PROTOCOL.md) · [위치·원천 감사](data/jhh_off_position_interaction/score/audit.tsv) · [Fold 지원](data/jhh_off_position_interaction/score/fold_support.tsv)

| 평가 지원 | 차이 0 대비 RMSE 증가 % [99% 구간] | 차이 0 대비 pooled R² 변화 %p [99% 구간] |
|---|---:|---:|
| 공통 86군, fold·role 중심화 | 0.130 [0.010, 0.265] | −0.324 [−0.688, −0.004] |
| 원천과 차량이 동일한 84군, 같은 중심화 | 0.148 [0.031, 0.280] | −0.357 [−0.715, −0.033] |

평균 차이를 제거한 86군에서 두 세포주의 관측 벡터 간 평균 코사인은 0.276이었다. 관측 차이 RMS가 0.7315인 동안 후보 차이 RMS는 0.0233이었다. 최선 공개 VC의 pooled R² 역시 −0.248%였고, 후보의 pooled R²는 그보다 0.076%p 낮았다. 후보 대 부호 반전 및 다른 약물 차이의 99% 구간에는 0이 포함됐다. [차이 지표 원표](data/jhh_off_position_interaction/score/paired.tsv) · [공개 VC 최선](data/jhh_off_position_interaction/score/best.tsv) · [관측·예측 규모](data/jhh_off_position_interaction/score/similarity.tsv)

앞선 결과가 보여 준 우위는 각 세포주 안의 약물 반응 벡터와 약물 간 대비에 해당한다. 이번 분석의 주 판정은 미통과다. 두 세포주의 표적은 각각 한 plate이고 학습·보정 절차도 달랐다. 관측 차이에서 세포주와 plate의 영향을 분리하려면 독립 배양과 plate 반복이 필요하다. 평가 웰은 기존 B1에 노출됐고, 예약 자료 접근과 새 모델 적합은 없었다.

사전 명세 `ef208e1` 이후 첫 Slurm 작업 `31840`은 `h5py` 경로가 빠져 채점 전에 중단됐다. 기존 런타임 경로를 복원한 `31841`은 종료 코드 0으로 완료됐다. 두 세포주의 원 공개 비교 점수·구간을 재현한 뒤 차이를 계산했다. [실행 원장](data/jhh_off_position_interaction/attempts.tsv) · [채점 영수증](data/jhh_off_position_interaction/score/receipt.json)
