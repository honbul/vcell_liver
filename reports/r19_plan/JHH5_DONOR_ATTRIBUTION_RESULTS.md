# JHH-5에서 동일 약물 실측의 기여와 세포주 간 편차 보정

**동일 약물의 비간계열 실측 평균이 State 적응과 두 약물 대응 대조를 앞섰다.** 생물학적 맥락에 따라 편차가 큰 유전자의 평균을 축소한 새 후보도 원 평균과 원점 배율을 모두 앞서 등록 기준을 통과했다. CMap 2020의 실측 10 µM·24 h JHH-5 609군·615웰·6개 RNA plate에서 평가했다. 기존 JHH-5 개발 점수를 본 뒤 명세한 후속 결과다. [사전 명세](JHH5_DONOR_ATTRIBUTION_PROTOCOL.md) · [짝 채점 원표](data/jhh5_donor_attribution/score/paired.tsv)

| 후보와 비교군 | 상대 RMSE 감소 [99% 구간] | Δ pooled R², %p [99% 구간] |
|---|---:|---:|
| 원 실측 평균 대 무반응 | +3.09% [1.70, 4.60] | +10.41 [6.35, 14.66] |
| 원 실측 평균 대 State 1토큰 적응 | +4.04% [2.91, 5.31] | +11.09 [7.78, 14.60] |
| 원 실측 평균 대 같은 풀의 다른 약물 실측 | +7.60% [6.04, 9.29] | +18.98 [14.76, 23.65] |
| 원 실측 평균 대 적합군 전체 donor 평균 | +3.49% [2.25, 4.87] | +10.27 [6.66, 14.12] |
| 세포주 간 편차 축소 대 원 실측 평균 | **+0.645% [0.535, 0.771]** | **+0.953 [0.703, 1.276]** |
| 세포주 간 편차 축소 대 원점 배율 | **+0.595% [0.383, 0.820]** | **+1.286 [0.795, 1.812]** |

두 약물 대조에는 해당 JHH-5 처리값을 쓰지 않았다. 다른 약물 대조는 같은 평가 plate·fold·농도 안에서만 바꾸고, 세 DMSO 역할에서 같은 대응을 유지했다. 네 귀속 비교가 절대 대비 기준을 통과했다. 원 평균과 원점 배율 사이의 두 99% 구간은 0을 포함했다. [짝 채점 원표](data/jhh5_donor_attribution/score/paired.tsv)

새 후보는 같은 원천 웰에서 얻은 세포주별 평균 및 분산만 썼다. JHH-5 적합 라벨 없이 RMSE **0.53486**, pooled R² **11.36%**를 얻었다. 두 양의 99% 구간과 0.5%·0.5%p 점추정 기준을 원 평균·배율 각각에서 통과했다. 여섯 평가 plate를 각각 빼도 두 지표는 두 기준선보다 좋았다. 배율 대비 상대 RMSE 개선은 0.39–0.70%로 plate 삭제에 따라 실용 문턱을 밑도는 경우가 있었다. Scaffold 재표집에서 원 평균 대비 상대 RMSE 구간 0.52–0.77%, R² 구간 +0.68–1.27%p였다. donor 세포주가 16개 이상인 605군에서도 두 방향이 유지됐다. 16개 미만은 4군이다. [Scaffold 재표집](data/jhh5_donor_attribution/score/scaffold.tsv) · [Plate 삭제](data/jhh5_donor_attribution/score/leave_plate.tsv) · [Donor 수 층](data/jhh5_donor_attribution/score/donor_count.tsv)

동일 plate·농도 안의 중심 대비에서도 새 후보는 원 평균 대비 RMSE 0.616% 감소 [99% 구간 0.504, 0.741], R² +0.911%p [0.666, 1.229]였다. Top-1은 원 평균 24.00%, 새 후보 24.03%, State 9.52%였다. 원 평균과 새 후보의 Top-1·Top-5 구간에 0이 포함됐다. 변화는 이 평가의 벡터 오차에서 확인됐다. [풀 대비](data/jhh5_donor_attribution/score/within_plate.tsv) · [검색](data/jhh5_donor_attribution/score/retrieval.tsv)

이 결과는 **같은 약물·용량·시점의 다른 세포주 실측이 있는 JHH-5 개발 조건**을 지지한다. 대상은 이전 B1 개발 명부에 있었고, 대부분의 화합물에 원천 세포주 16개 이상이 있었다. 데이터가 적은 약물과 다른 성인 HCC에서의 확인은 별도로 필요하다. 기존 저랭크 판정은 미통과다. [JHH-5 지원](HCC_LINCS_24H_SUPPORT.md) · [원 비교](HCC_LINCS_24H_TRANSFER_RESULTS.md)

명세 `521b0f9`, 병렬 예측 `31629` → 동결 `31630` → 별도 채점 `31631`이다. 종료 코드 0인 다섯 작업에 CPU 0.024시간이 할당됐고 GPU는 0개였다. 기존 donor 평균은 배열 원소까지 동일하게 재현됐다. 새 동결과 원 동결, 역할 영수증, 채점 파일의 SHA-256을 대조했다. 새 다운로드와 예약 발현 접근은 없었다. [작업 원장](data/jhh5_donor_attribution/jobs.tsv) · [동결](data/jhh5_donor_attribution/freeze.json) · [채점 영수증](data/jhh5_donor_attribution/score/receipt.json) · [자원 사용](data/jhh5_donor_attribution/resource_usage.tsv)
