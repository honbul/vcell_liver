# 동결 후보의 parent·plate 기여도 분석

이미 점수가 공개된 T4와 S/L B의 사후 기술 분석이다. T4는 132 parent·268 물리 웰·12 RNA plate에서 `operator6` 대 `scalar6`, `identity6`, `direct24`다. S/L B는 493 parent·2,324 물리 웰·41 RNA plate에서 `B_R17` 대 `B_blend_anchor`다. 각 결과의 동결 해시와 전체 지원에서 원 점수와 99% 구간을 대조한 뒤 같은 역할ㆍ유전자ㆍ대상만 분석한다. 새 예측, 모델 설정, 데이터 적합은 없다.

- 양의 이득은 기준선에서 후보의 parent-macro RMSE를 뺀 값과 후보의 pooled R²에서 기준선을 뺀 값이다. Parent별 RMSE 감소와 SSE 감소에서 각각 양의 기여를 순위화한다. 상위 5·10·20 parent가 차지하는 양의 기여 비중을 낸다. 각 순위의 상위군을 빼고 두 지표를 재집계한다. 동률이면 canonical parent 문자열 순이다. 두 지표의 제거 목록은 해당 지표에서 정한다. 절대·상대 변화의 기존 명세와 비교군은 그대로 유지한다.
- 평가 plate 하나씩을 빼고 같은 두 지표의 점추정을 재집계한다. 학습된 예측은 그대로 사용하며 새 plate 교차검증으로 해석하지 않는다.
- 물리 plate와 canonical parent가 교차하는 지원을 점검하기 위해 양쪽에 서로 독립인 Exp(1) 가중치를 주고 곱으로 결과를 재가중한다. 표본별 대응, 같은 plate와 parent에 반복된 세 vehicle 역할 및 물리 웰은 같은 가중치를 공유한다. Parent-macro RMSE의 parent 안 평균은 plate 가중치로, parent 사이 평균은 parent 가중치로 계산한다. Pooled R²는 곱 가중치다. seed 150104, 가중치 10,000회, 0.5 및 99.5 분위수로 고정한다. 이 구간은 추가 의존성 민감도이며 원 parent bootstrap의 사전 판정을 보존한다.

Score-only 재계산에서 승인된 기존 DEV 입력과 동결 배열만 읽는다. 코호트별 원표, 상위군 제거, plate 제거, parent·plate 구간 및 자원 기록을 남긴다.
