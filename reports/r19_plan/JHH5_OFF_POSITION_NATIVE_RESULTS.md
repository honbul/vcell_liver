# JHH-5 6개 plate의 위치 분리 공개 VC 비교

**배율 보정 편차 축소는 최선 공개 VC 대비 사전 기술 기준을 통과했다.** CMap JHH-5 10 µM·24 h의 기존 305개 화합물군과 309개 처리 웰에서 상대 RMSE가 **0.751% [99% 구간 0.558, 0.974]** 낮았고, pooled R²는 **1.662%p [1.154, 2.224]** 높았다. 두 지표 모두 3개 fold에서 개선됐다. 이 후속 비교의 평가 웰은 모두 B1에 사용됐다. [채점 전 명세](JHH5_OFF_POSITION_NATIVE_PROTOCOL.md) · [최선 대비 원표](data/jhh5_off_position_native/score/best.tsv) · [판정](data/jhh5_off_position_native/gate.json)

| 평가지표 | 배율 보정 편차 축소 | PRnet·chemCPA·TranSiGen 및 단일 배율 | XPert | 8토큰 State |
|---|---:|---:|---:|---:|
| Parent-macro RMSE | **0.55718** | 0.56140 | 0.56391 | 0.57680 |
| Pooled R² | **7.612%** | 5.950% | 5.044% | −1.134% |

같은 8개 비간계열 약물 실측을 사용한 PRnet·chemCPA·TranSiGen은 54개 분할 전체에서 잔차 배율 0을 골라 공통 단일 배율과 일치했다. XPert는 45개 분할에서 양의 배율을 골랐으며, 전체 평가에서 XPert는 단일 배율보다 RMSE가 높고 R²가 낮았다. 두 지표의 10,000회 재표집에서 항상 단일 배율과 일치한 세 모델이 공동 최선이었다. 동점일 때 빈도표에는 등록 순서의 PRnet을 적었다. 새 비교군 추가 후 개선 크기가 유지된 이유다. [개별 점수와 차이](data/jhh5_off_position_native/score/individual.tsv) · [설정 선택](data/jhh5_off_position_native/score/selection.tsv) · [최선 선택 빈도](data/jhh5_off_position_native/score/winner_frequency.tsv)

6개 평가 plate를 각각 제외해도 두 지표의 99% 구간이 개선 방향을 유지했다. 가장 작은 이득은 `ASG002` 제외 시였으며, 220군에서 RMSE 0.548% [0.431, 0.672], R² 0.991%p [0.709, 1.303]였다. 기존 B0 비교에서 같은 plate를 빼면 구간이 0을 포함했던 결과와 함께 해석해야 한다. 두 세포주의 공개 VC 비교는 CMap의 기존 노출 관측에 근거한다. 독립 배양에서의 재현성은 아직 실측해야 한다. [Plate 제외 원표](data/jhh5_off_position_native/score/leave_plate.tsv) · [기존 B0 민감도](JHH5_OFF_POSITION_RESULTS.md) · [JHH-7 확장 비교](JHH7_OFF_POSITION_NATIVE_RESULTS.md) · [두 세포주 비교 그림](manuscript/figure3.pdf)

평가 위치·plate·fold·scaffold 분리와 서로소 DMSO 역할을 유지했다. 내부 적합 96–148군, 선택 24–36군을 쓰고 122–183군으로 재적합했다. 54개 원 단일 배율 배열을 정확히 재현했다. 216개 새 예측과 기존 후보·State를 동결해 별도 채점했으며 기존 후보·단일 배율·State·B0 점수를 대조했다. 명세 `2642719`, 준비 `31775`, 예측 `31776`, 동결 `31777`, 채점 `31778`은 모두 종료 코드 0이었다. 예측 프로세스의 GPU 사용 시간 합계는 약 0.40시간이다. R18 예약 발현을 읽지 않았다. [동결 영수증](data/jhh5_off_position_native/freeze.json) · [채점 영수증](data/jhh5_off_position_native/score/receipt.json) · [실행 원장](data/jhh5_off_position_native/jobs.tsv) · [자원 원표](data/jhh5_off_position_native/resource_usage.tsv)
