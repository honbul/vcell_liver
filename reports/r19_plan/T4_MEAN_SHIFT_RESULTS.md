# HEPG2 시간 전이에서 평균 이동과 대각 보정의 효과

같은 6시간 실측을 받은 rank-16 연산자는 평균 이동과 공통 배율을 함께 적합한 대조군과 유전자별 대각 ridge를 모두 앞섰다. 132개 화합물, 121개 Bemis–Murcko scaffold, 12개 물리 RNA plate의 24시간 웰 268개와 세 DMSO 역할을 이전 scaffold 분석에서 그대로 재사용했다. [등록 명세](T4_MEAN_SHIFT_PROTOCOL.md)

| 대조군 | 대조군 RMSE | 대조군 pooled R² | Rank-16 상대 RMSE 개선 [paired 99% CI] | Rank-16 pooled R² 개선 [paired 99% CI] |
|---|---:|---:|---:|---:|
| 평균 이동 + 공통 배율 | 0.94723 | 20.502% | +4.164% [2.709, 5.656] | +5.428%p [2.984, 7.918] |
| 유전자별 대각 ridge | 0.93957 | 21.610% | +3.382% [2.081, 4.731] | +4.320%p [2.116, 6.590] |

Rank-16의 RMSE는 0.90779, pooled R²는 25.930%였다. 10,000회 whole-parent 짝 재표집에서 두 비교의 두 지표 모두 개선 하한이 0을 넘었다. Scaffold 군집으로 재표집한 99% 구간도 같은 방향이었다. 평균 이동 + 공통 배율 대비 RMSE [2.769, 5.648]%, pooled R² [2.959, 8.043]%p, 대각 ridge 대비 RMSE [2.123, 4.729]%, pooled R² [2.111, 6.668]%p였다. 기존 원점 배율에 대한 RMSE 개선은 7.24%였으며 이보다 작은 새 추정치가 평균 이동을 통제한 뒤 남는 이득이다. [채점 원표](data/t4_mean_shift/score/paired.tsv) · [이전 scaffold 분석](T4_SCAFFOLD_RESULTS.md)

이 결과는 동일 HEPG2·L1000 조건에서 평균 이동과 유전자별 독립 배율을 넘어서는 유전자 간 저랭크 시간 변환의 추가 예측 가치를 지지한다. PHH 및 HepaRG 결과는 이 결론의 세포·assay 범위를 제한한다. [PHH 전이](T4_CONTEXT_TRANSFER_RESULTS.md) · [HepaRG 비교](T4_CROSS_ASSAY_RESULTS.md)

코드 `5468802`를 고정하고 G1 CPU 예측 `32412` → 동결 `32413` → 별도 채점 `32414`를 실행했다. [작업 원장](data/t4_mean_shift/jobs.tsv) · [적합 지원](data/t4_mean_shift/score/fit_support.tsv) · [동결 영수증](data/t4_mean_shift/freeze.json) · [채점 영수증](data/t4_mean_shift/score/receipt.json)
