# T4 세포 맥락 전이의 절편과 잔차 검증

기존 L1000 HEPG2→PHH 연산자는 같은 source에서 적합한 배율보다 오차가 컸다. 적합 세포의 평균을 더하는 효과와 낮은 순위의 행렬을 적용하는 효과를 구분하고, 보정하지 않는 방향을 유지하는 잔차 모델을 검정한다. 기존 두 DEV 집합의 고정 코호트이므로 이 후속 결과는 독립 외부 확인으로 해석하지 않는다.

HEPG2→PHH 67군·105웰이 주 비교다. PHH→HEPG2 132군·268웰은 역방향 기술 비교다. 기존 T4 맥락 전이의 각 fold·역할 source 적합 합집합과 평가 fold 전체 제외, 대상 plate 제외, 고정 rank 16과 ridge 1을 유지한다. 대상 세포의 실제 6 h 벡터 x를 동일하게 제공한다. 모든 새 설정과 적합은 source에서만 정하고 target 24 h 처리값과 score DMSO는 동결 뒤 별도 채점에만 쓴다.

| 예측 | 규칙 |
|---|---|
| `cross_scalar` | 기존 동결 `a·x`. a는 source의 parent 균등 최소제곱 0–2 배율이다. |
| `cross_operator` | 기존 동결 `b + xW`. b는 source 24 h 평균에서 source 6 h 평균의 W 예측을 뺀 값이다. |
| `scalar_with_offset` | `a·x + b`. 기존 연산자의 절편만 배율 예측에 더한다. |
| `operator_no_offset` | `xW`. 기존 연산자에서 절편만 제거한다. |
| `residual_only` | `a·x + xWᵣ`. source에서 `y − a·x`를 같은 rank·ridge로 적합하고, 계산한 가중치 Wᵣ만 배율 위에 더한다. |
| `residual_with_offset` | `a·x + bᵣ + xWᵣ`. 잔차 모델의 source 절편까지 추가하는 기술 대조다. |

원 전이 배율·연산자는 동일 입력에서 최대 절대차 1e−6 이내로 재현한 뒤 원 동결값을 비교군으로 사용한다. 절편 검증의 짝 비교는 `cross_operator`와 `operator_no_offset`, `scalar_with_offset`과 `cross_scalar`, `residual_with_offset`과 `residual_only`다. 주 후보 `residual_only`는 `cross_scalar`와 `cross_operator` 각각에 대해 RMSE 차이 99% 상한 <0, pooled R² 차이 99% 하한 >0을 만족해야 한다. 두 비교에서 RMSE 0.5% 개선과 R² +0.5%p도 함께 평가한다. 대상 24 h 개발 적합 자료를 사용한 `local_scalar`는 별도 기술 참조다. 두 방향 모두 10,000회 짝 parent 재표집이며 새 rank·배율 탐색은 없다.

명세 → 예측 → 동결 → 별도 채점을 Slurm 의존성으로 실행한다. 기존 T4 잠금 명부의 허용된 측정만 읽는다. 신규 다운로드 및 R18 예약 발현 접근은 없다.
