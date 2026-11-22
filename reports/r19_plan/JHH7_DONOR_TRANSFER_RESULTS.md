# JHH-7의 동일 약물 실측 전이 기술 평가

**JHH-7에서도 같은 약물의 비간계열 실측 평균이 무반응·State·다른 약물 실측 대조를 앞섰다.** JHH-5에서 추가 이득을 보인 세포주 간 편차 축소는 JHH-7의 두 기술 재현 기준에 미달했다. CMap 2020 실측 10 µM·24시간 JHH-7의 99군·99웰·한 RNA plate에 대한 결과다. [사전 명세](JHH7_DONOR_TRANSFER_PROTOCOL.md) · [짝 채점 원표](data/jhh7_donor_transfer/score/paired.tsv)

| 후보와 비교군 | 상대 RMSE 감소 [99% 구간] | Δ pooled R², %p [99% 구간] |
|---|---:|---:|
| 동일 약물 평균 대 무반응 | **12.13% [7.59, 16.40]** | **+28.91 [19.35, 36.87]** |
| 동일 약물 평균 대 JHH-5 적합 State | **13.71% [9.82, 17.49]** | **+31.38 [23.31, 38.51]** |
| 동일 약물 평균 대 같은 풀의 다른 약물 | **23.34% [17.99, 28.47]** | **+54.65 [40.70, 72.00]** |
| 편차 축소 대 동일 약물 평균 | +0.260% [0.055, 0.491] | +0.301 [−0.068, +0.760] |
| 편차 축소 대 JHH-5 적합 원점 배율 | +0.944% [−0.230, +2.075] | +2.032 [−0.075, +3.719] |

동일 약물 평균의 RMSE는 0.63609, R²는 28.91%였다. 편차 축소의 점추정은 RMSE 0.63444, R² 29.21%였다. 원 평균 대비 상대 RMSE의 99% 상한 0.491%는 사전 실용치 0.5% 미만이었고 R² 구간은 0을 포함했다. 배율 대비는 두 구간이 0을 포함했다. Scaffold 재표집에서도 판정은 같았다. [Scaffold 구간](data/jhh7_donor_transfer/score/scaffold.tsv)

동일 plate·농도 대비에서도 원 평균은 State 대비 RMSE를 13.91% [10.26, 17.62] 줄였고 R²를 30.00%p [21.94, 37.14] 높였다. 편차 축소 대 원 평균의 두 99% 구간에는 0이 포함됐다. 원 평균과 편차 축소의 Top-1은 각각 49.83%와 50.17%, State는 18.52%, 무작위 기대는 3.03%였다. [풀 대비](data/jhh7_donor_transfer/score/centered.tsv) · [검색](data/jhh7_donor_transfer/score/retrieval.tsv)

비교군 State와 배율은 JHH-7 평가 99개 parent 및 같은 scaffold를 제거한 JHH-5 475군에 각각 적합했다. 후보 식은 이전 단계의 식을 고정했고 JHH-7 처리값은 채점에만 썼다. 평가 99군 중 95개 parent는 JHH-5 개발 지원과 겹친다. JHH-7의 물리 웰은 이전 B1에도 포함됐고 한 plate의 13개 DMSO를 세 역할로 나눴다. 결과는 세포주 간 조건부 전이를 보여 주며 plate 간 변동과 독립 성인 HCC 확인은 남아 있다. [명부와 적합 지원](data/jhh7_donor_transfer/access/support.tsv) · [JHH-5 귀속](JHH5_DONOR_ATTRIBUTION_RESULTS.md)

명세 `a7c2fb8`과 접근 명부 `f9873d4`를 고정했다. 메타데이터 잠금 `31634` → 세 역할 예측 `31635` → 동결 `31636` → 별도 채점 `31637`의 여섯 작업은 종료 코드 0이었다. GPU 할당 0.029시간, 동시 사용 최대 2개였다. 세 역할의 다른 약물 대응과 동결·채점 SHA-256을 대조했다. 새 다운로드와 예약 발현 접근은 없었다. [실행 원장](data/jhh7_donor_transfer/jobs.tsv) · [동결](data/jhh7_donor_transfer/freeze.json) · [영수증](data/jhh7_donor_transfer/score/receipt.json)
