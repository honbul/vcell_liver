# JHH-5 성인 간암 24시간 약물 반응 전이 결과

**저랭크 보정은 JHH-5에서 사전 통과 기준을 충족하지 못했다.** CMap 2020의 실측 10 µM·24시간 JHH-5 609군·615웰·6개 RNA plate에서 평가했다. 모든 전이 모델에는 같은 약물을 비간계열에서 측정한 벡터를 제공했다. 각 조합에서 해당 평가 plate 전체와 평가 fold의 parent·scaffold를 적합에서 제외했다. [사전 명세](HCC_LINCS_24H_TRANSFER_PROTOCOL.md) · [지원과 노출](HCC_LINCS_24H_SUPPORT.md) · [원표](data/hcc_lincs24/transfer/score/paired.tsv)

| 같은 평가 웰의 예측 | Parent-macro RMSE | Pooled R² |
|---|---:|---:|
| 원점 배율을 적용한 동일 약물 실측 | **0.53806** | 10.08% |
| 동일 약물 비간계열 실측 평균 | 0.53833 | **10.41%** |
| 평균 보정 배율 | 0.54832 | 7.23% |
| 무반응 B0 | 0.55550 | 0.00% |
| 평균 보정 배율 + rank-16 잔차 | 0.55964 | 2.26% |
| 같은 anchor + 잔차 parent 순열 | 0.55961 | 2.34% |
| State Transition 1토큰 적응 | 0.56100 | −0.68% |

| 사전 비교, rank-16 잔차 결합의 상대 효과 | 상대 RMSE 감소 [99% 구간] | Δ pooled R², %p [99% 구간] |
|---|---:|---:|
| 평균 보정 배율 | **−2.07% [−2.99, −1.25]** | **−4.97 [−7.48, −2.66]** |
| State 적응 | +0.24% [−0.65, +1.17] | +2.94 [+0.53, +5.32] |
| 잔차 parent 순열 | −0.006% [−0.562, +0.550] | −0.075 [−1.633, +1.455] |

원점 배율과 직접 실측 평균이 각각 RMSE, R²에서 가장 좋은 점추정이었다. 평균 보정 배율에 잔차를 추가하면 두 지표가 악화했고, 여섯 plate를 하나씩 빼도 이 방향이 유지됐다. Scaffold 재표집에서도 평균 보정 배율 대비 대표 두 지표가 악화했다. 순열과의 차이 구간에 0이 포함돼, 등록한 잔차 사상의 약물별 추가 이득은 확인되지 않았다. [Scaffold 구간](data/hcc_lincs24/transfer/score/scaffold.tsv) · [Plate 삭제](data/hcc_lincs24/transfer/score/leave_plate.tsv)

동일 plate·농도 풀 대비를 기술적으로 재채점하면 잔차 결합은 State 적응보다 RMSE가 0.83% 낮고 [99% 구간 0.03, 1.65], R²가 +2.91%p 높았다 [0.79, 5.16]. 평균 보정 배율보다 RMSE는 2.54% 높았고 [1.63, 3.56], R²는 −6.00%p였다 [−8.91, −3.38]. Top-1은 평균 보정 배율 24.00%, 잔차 결합 21.10%, State 9.52%, 무작위 기대 2.93%였다. State 비교는 978유전자 **24시간 비간계열 실측을 한 토큰으로 쓴 새 적합**이다. 원표 키 `State_6h`는 재사용한 모형 코드의 이름이다. [풀 대비](data/hcc_lincs24/transfer/score/within_plate.tsv) · [검색](data/hcc_lincs24/transfer/score/retrieval.tsv)

JHH-5 대상 웰 모두는 과거 B1의 개발 명부에 있었다. 이번 적합의 조합당 지원은 267–332군이며, 세 역할에 할당된 615개 평가 웰과 fit/score 분리를 별도로 검산했다. 독립 plate와 다른 위치를 모두 갖는 대상 반복은 6군이므로 모집단 반복 상한은 판단하지 않았다. 이 비교에서 직접 실측의 점추정은 유망하지만, 등록 후보 승격과 성인 HCC 독립 확인은 별개다. [적합 원표](data/hcc_lincs24/transfer/score/fit_support.tsv)

명세 `ef37777`, 세포주×RNA plate DMSO 배정 수정 `3e5b984`, 명부 잠금 `b039cfe`다. 최초 명부 작업 `31613`은 한 donor RNA plate에 복수 세포주가 있어 DMSO 검사를 통과하지 못했다. 세포주별 DMSO를 분리한 명부 `31614` → donor `31615` → 예측 `31616` → 동결 `31617` → 별도 채점 `31618`이 종료 코드 0이었다. GPU 0.081시간, 동시 최대 2개다. 동결 해시 및 54개 조합 영수증을 대조했으며 R18 예약 발현 접근과 새 다운로드는 없었다. [실행 원장](data/hcc_lincs24/transfer_jobs.tsv) · [동결](data/hcc_lincs24/transfer/freeze.json) · [채점 영수증](data/hcc_lincs24/transfer/score/receipt.json) · [자원 사용](data/hcc_lincs24/transfer/resource_usage.tsv)
