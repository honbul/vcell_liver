# Control 평균화의 작은 감도 분석

새 학습 없이 기존 native-common191개 test target/12839cells/primary1977genes를 사용했다. 전체 same-batch control 평균과 각 treated cell에 이미 고정된 sampled control ID의 평균을 비교한다. 분석 규칙은 `configs/research/control_sensitivity_v3.json`에 실행 전 고정했다. 기존 모델·선택 기록·예측·metric은 바꾸지 않았다.

| 예측기 | 원 full-control 평균 입력 RMSE | Recorded sampled-control 평균 입력 RMSE | 20개 control resampling의 평균±SD |
|---|---:|---:|---:|
| No change | .126950 | .145649 | .145743±.000521 |
| Training mean Delta | .120503 | .139054 | .138877±.000492 |
| GO/PCA/ridge | .110730 | .130501 | .130067±.000457 |

Simple predictor의 fitted Delta를 유지한 실제 입력 control 평균 교체다. Train target response PCA/ridge를 새로 맞추지 않았다. 원 native scGPT pretrained .123941, random .122679와 GEARS .137319는 sampled-control 입력에서 나온 예측이다. 따라서 원 simple-vs-native 순위는 control 평균화 차이를 포함한다. 이 민감도로 모든 모델의 동일 estimator 재학습 비교를 대신하지 않는다. Pretrained-vs-random은 처음부터 같은 입력·구조·optimizer를 맞춘 별도 대조다.

평가 reference만 바꾸는 경우에는 주의가 필요하다. 고정 예측과 truth에서 같은 control을 빼면 RMSE는 대수적으로 동일하지만 Delta Pearson은 달라진다. 예를 들어 full-control no-change 예측을 유지하고 sampled-control reference를 빼면 Delta Pearson .49474가 생긴다. 이는 새로운 response 예측 정보가 생긴 증거가 아니다. 공통 control noise가 두 Delta에 함께 들어간다. 원 reporting reference는 full matched-control 평균으로 유지한다.

Native 입력 감도는 점수와 무관하게 사전 고정한 첫 test target AAR2의38cells에서 확인했다. 같은 저장 checkpoint를 읽고 sampled-control expression 대신 같은 배치의 full-control mean expression을 넣었다. 출력 평균을 원 full-control evaluator에서 비교한 RMSE는 pretrained .108824→.108399, random .104953→.105519였다. 원 aggregate prediction과 작은 replay의 max absolute discrepancy는 각각 .000103/.000206이며 A30·batch 구성 차이를 함께 기록했다. 이 한 target 결과로 전체191target 순위를 바꾸거나 일반화하지 않는다. 새 optimizer update는 없다.

선택 지표는 원 validation primary1977-gene intervention-mean MSE다. 보고 지표는 test intervention별 RMSE/Delta Pearson과 원 top20 보조 지표다. Top20은 evaluator-only absolute observed mean Delta 순위이며 FDR 기반 DEG 검증이 아니다. Control resampling20seed의 표준편차도 biological replicate 불확실성이 아니다. 원 biological replicate assignment는 unknown이다.

실제 결과·입력/모델 hash는 `control_sensitivity/aggregation_provenance.json`, `native_first_target.json` 및 같은 폴더의 TSV에 있다.
