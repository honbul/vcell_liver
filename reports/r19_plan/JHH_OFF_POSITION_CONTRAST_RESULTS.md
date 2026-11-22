# 성인 HCC 10 µM 위치 분리 예측의 약물 대비와 검색

**같은 평가 풀의 평균 반응을 뺀 뒤에도 공개 VC 최선 대비 벡터 예측의 개선이 남았다.** JHH-5는 기존 305군·309웰, JHH-7은 88군·88웰을 그대로 평가했다. 풀은 RNA plate, 평가 fold, DMSO 역할과 실측 10 µM으로 고정했다. JHH-5의 54개 풀에는 4–32군, JHH-7의 9개 풀에는 27–32군이 있었다. 두 코호트 모두 기존 B1에 노출됐다. [채점 전 명세](JHH_OFF_POSITION_CONTRAST_PROTOCOL.md) · [JHH-5 원표](data/jhh_off_position_contrast/JHH5/centered_best.tsv) · [JHH-7 원표](data/jhh_off_position_contrast/JHH7/centered_best.tsv)

| 대상 | 최선 공개 VC 대비 중심화 RMSE 감소 % [99% 구간] | Pooled R² 증가 %p [99% 구간] |
|---|---:|---:|
| JHH-5 | **0.799 [0.607, 1.019]** | **1.627 [1.106, 2.203]** |
| JHH-7 | **1.338 [0.741, 2.068]** | **2.838 [1.421, 4.372]** |

두 코호트가 등록한 중심화 오차와 실용 점추정 기준을 통과했다. 같은 풀의 다른 약물 원천으로 바꾼 대조보다 중심화 RMSE가 JHH-5에서 5.694% [4.268, 7.204], JHH-7에서 10.985% [7.928, 14.325] 낮았다. 같은 약물의 원천이 풀 공통 반응을 넘어 약물 간 차이를 담고 있음을 지지한다. [JHH-5 대조군 원표](data/jhh_off_position_contrast/JHH5/centered.tsv) · [JHH-7 대조군 원표](data/jhh_off_position_contrast/JHH7/centered.tsv)

| 대상 | Top-1 편차 축소 | Top-1 최선 공개 VC | Top-1 무작위 기대 | 최선 공개 VC 대비 Top-1 차이 %p [99% 구간] |
|---|---:|---:|---:|---:|
| JHH-5 | 24.37% | 23.83% | 5.83% | +0.55 [−1.09, +2.30] |
| JHH-7 | 40.53% | 39.77% | 3.41% | +0.76 [−1.52, +3.41] |

편차 축소의 Top-1과 Top-5는 두 세포주에서 State와 다른 약물 대조를 앞섰다. 최선 공개 VC와 비교한 두 검색률의 99% 구간은 모두 0을 포함했다. 중심화 벡터 코사인은 최선 공개 VC보다 각각 0.0075와 0.0137 높았다. 검색 후보 수는 두 세포주에서 다르며, Top-5는 후보 6군 이상의 풀로 한정해 JHH-5 301군과 JHH-7 88군에서 계산했다. [JHH-5 짝 검색](data/jhh_off_position_contrast/JHH5/retrieval_paired.tsv) · [JHH-7 짝 검색](data/jhh_off_position_contrast/JHH7/retrieval_paired.tsv)

명세 `6a2484b` 이후 첫 작업 `31834`는 공통 벡터 로더가 단일 배율 키를 넘기지 않아 출력 전에 중단됐다. 로더를 `9eab4ab`에서 수정한 뒤 작업 `31836`에서 두 세포주 모두 종료 코드 0이었다. 최종 채점은 원 절대 점수와 구간을 먼저 재현했다. 예측 재적합과 예약 자료 접근은 없었다. [실행 원장](data/jhh_off_position_contrast/launch.tsv) · [JHH-5 영수증](data/jhh_off_position_contrast/JHH5/receipt.json) · [JHH-7 영수증](data/jhh_off_position_contrast/JHH7/receipt.json)
