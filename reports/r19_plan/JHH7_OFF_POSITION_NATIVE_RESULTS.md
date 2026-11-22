# JHH-7에서 다섯 공개 VC를 포함한 위치 분리 기술 비교

**배율 보정 편차 축소는 확장한 최선 VC 대비에서도 사전 기술 기준을 통과했다.** JHH-7 실측 10 µM·24 h의 88군을 그대로 썼다. 새로 적합한 네 VC와 기존 동결 State의 최선을 재표집마다 지표별로 다시 골랐다. 후보의 상대 RMSE 감소는 **1.278% [99% 구간 0.680, 2.005]**, pooled R² 증가는 **2.945%p [1.574, 4.366]**였다. 두 지표의 개선 방향은 세 평가 fold 모두에서 같았다. [채점 전 명세](JHH7_OFF_POSITION_NATIVE_PROTOCOL.md) · [최선 대비 원표](data/jhh7_off_position_native/score/best.tsv) · [판정](data/jhh7_off_position_native/gate.json)

| 평가 arm | Parent-macro RMSE | Pooled R² |
|---|---:|---:|
| 배율 보정 편차 축소 | **0.66293** | **16.447%** |
| PRnet·chemCPA·TranSiGen 적응 및 단일 배율 | 0.67151 | 13.503% |
| XPert 적응 | 0.67212 | 13.432% |
| 8토큰 State 적응 | 0.72211 | −1.661% |
| 무반응 B0 | 0.70894 | 0.000% |

PRnet·chemCPA·TranSiGen은 각 세 회전의 JHH-5 내부 선택에서 모두 신경망 잔차 배율을 0으로 골랐다. 아홉 예측 배열이 원 동결 단일 배율과 정확히 일치했다. XPert는 순서대로 0.25·0.125·1의 잔차 배율을 골랐고 평가에서 단일 배율보다 RMSE가 소폭 높았다. RMSE 재표집의 최선은 항상 세 동점 모델에 해당했다. Pooled R²에서는 10,000회 중 407회에 XPert가 최선이었다. 동점일 때 표에는 등록 순서상 첫 모델을 기록했다. [개별 비교](data/jhh7_off_position_native/score/individual.tsv) · [선택표](data/jhh7_off_position_native/score/selection.tsv) · [최선 빈도](data/jhh7_off_position_native/score/winner_frequency.tsv)

네 VC는 같은 원천 8개에서 만든 단일 배율 anchor와 각 구현의 구조 특징, JHH-5 적합 DMSO의 고정 참조를 받았다. 내부 적합 163군과 선택 40군의 DMSO 역할은 서로소였다. 최종 적합은 화합물·scaffold를 제외한 JHH-5 203군만 사용했다. JHH-7은 이미 B1에 사용된 단일 평가 plate이며, R18 예약 발현이나 새 독립 파일럿 측정은 사용하지 않았다. 이 결과는 지정 정보 계약 아래의 새로 적합한 공개 모델 적응을 비교한다. [원 위치 분리 기술 평가](JHH7_OFF_POSITION_RESULTS.md)

명세 `1b42e94`, 준비 `31741` → 예측 `31742` → 동결 `31743` → 별도 채점 `31744`가 종료 코드 0이었다. 12개 새 예측 해시, 기존 후보·State 원표 점수 재현, JHH-5 원천과 참조 유전자 명부를 대조했다. [실행 원장](data/jhh7_off_position_native/jobs.tsv) · [동결](data/jhh7_off_position_native/freeze.json) · [채점 영수증](data/jhh7_off_position_native/score/receipt.json)
