# JHH-5 위치 분리 보정의 JHH-7 10 µM 전이

**JHH-7에서도 배율을 맞춘 편차 축소가 사전 공동 기술 기준을 통과했다.** JHH-5 10 µM에서만 적합하고 JHH-7 10 µM·24 h의 88군을 평가했다. 평가와 다른 RNA plate·웰 인덱스에서 얻은 비간계열 원천 8개를 모든 비교군에 동일하게 제공했다. State는 8개의 개별 벡터를 받았다. [사전 명세](JHH7_OFF_POSITION_PROTOCOL.md) · [짝 채점 원표](data/jhh7_off_position/score/paired.tsv)

| 배율을 맞춘 편차 축소의 비교군 | 상대 RMSE 감소 % [99% 구간] | Pooled R² 개선 %p [99% 구간] | 두 지표 개선 fold |
|---|---:|---:|---:|
| 같은 8개 원천의 단일 배율 | +1.278 [+0.680, +2.005] | +2.945 [+1.574, +4.381] | 3/3 |
| 같은 8개 벡터를 받은 State | +8.195 [+5.984, +10.504] | +18.109 [+12.085, +23.394] | 3/3 |
| 같은 보정을 적용한 다른 약물 | +9.244 [+6.615, +12.176] | +20.037 [+13.415, +26.825] | 3/3 |
| 무반응 B0 | +6.491 [+4.027, +9.102] | +16.447 [+10.139, +22.502] | 3/3 |

네 비교 모두 상대 RMSE 0.5%·R² 0.5%p 이상의 점추정, 99% 구간의 개선 방향, fold 3개 중 2개 이상에서 두 지표 개선이라는 기준을 충족했다. 일반 단일 배율도 State·다른 약물·B0를 대상으로 한 별도 기준을 통과했다. 배율 없는 편차 축소는 절대 RMSE 0.65710으로 배율을 맞춘 후보 0.66293보다 낮았다. 이 비교와 무보정 원천 평균 비교에서는 두 지표의 99% 구간이 0을 포함했다. 따라서 JHH-5의 보정식을 고정해도 동등 입력의 단일 배율과 State를 앞선다는 근거는 얻었지만, JHH-7에서 배율 적용 자체의 추가 가치는 구분되지 않았다. [판정 원표](data/jhh7_off_position/gate.json) · [Fold 원표](data/jhh7_off_position/score/folds.tsv) · [배율](data/jhh7_off_position/score/fits.tsv)

원 JHH-7 전체 99군과 해당 scaffold를 제외하자 JHH-5 적합은 203군·205웰·6개 RNA plate였다. JHH-7의 88웰은 모두 B1에 사용됐고, 86군은 앞선 JHH-5 개발 명부에도 있었다. 이번 적합은 그 겹치는 화합물과 scaffold를 제외했다. 평가는 한 RNA plate와 13개의 DMSO에 한정됐다. 같은 원천 plate의 평가 위치에서 다른 약물까지 짝지을 수 있는 세포주와 plate가 8개인 경우는 12군이었다. 주 다른 약물 대조는 평가 fold 내 순환이다. [지원 영수증](data/jhh7_off_position/access/receipt.json) · [접근 검산](data/jhh7_off_position/access/verified.json)

명세 `f66f879`, 접근 `31724` → 예측 `31725` → 동결 `31726` → 별도 채점 `31727`이 종료 코드 0이었다. 세 역할의 예측 해시와 원표를 검증했다. 예약 R18 발현에는 접근하지 않았다. [실행 원장](data/jhh7_off_position/jobs.tsv) · [동결](data/jhh7_off_position/freeze.json) · [채점 영수증](data/jhh7_off_position/score/receipt.json) · [두 세포주 비교 그림](manuscript/figure3.pdf)
