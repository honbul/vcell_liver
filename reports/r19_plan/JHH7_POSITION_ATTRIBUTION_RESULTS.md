# JHH-7 저용량의 약물·위치 분리와 동결 재채점

**JHH-7 1 µM에서 다른 위치의 동일 약물 실측이 위치를 맞춘 다른 약물 실측보다 정확했다.** 두 입력은 같은 U2OS 원천 plate·DMSO·짝 수를 공유했다. 동일 약물 모델은 무반응 B0보다 RMSE가 높았으므로 두 비교를 함께 요구한 사전 기준은 미통과다. 원 JHH-7 성능을 본 뒤 등록한 후속 개발 분석이다. [사전 명세](JHH7_POSITION_ATTRIBUTION_PROTOCOL.md) · [위치 지원](data/jhh7_position_attribution/support.json) · [판정](data/jhh7_position_attribution/gate.json)

| 농도와 공통 지원 | 다른 위치의 같은 약물과 비교한 대상 | 상대 RMSE 개선 % [99% 구간] | Pooled R² 변화 %p [99% 구간] |
|---|---|---:|---:|
| 1 µM, 80군 | 같은 위치의 다른 약물 | +3.14 [+0.39, +5.89] | +8.20 [+0.72, +15.16] |
| 1 µM, 80군 | B0 | −3.10 [−5.71, −0.77] | −3.93 [−11.31, +2.35] |
| 1 µM, 원천 plate 3개 이상 49군 | 같은 위치의 다른 약물 | +1.50 [+0.12, +2.98] | +3.60 [+0.54, +6.84] |
| 0.1 µM, 80군 | 같은 위치의 다른 약물 | +1.18 [−1.17, +3.38] | +3.28 [−2.92, +9.34] |

원 1 µM 원천 4,973웰 중 4,496웰(90.4%)이 목표와 행·열 인덱스를 공유했다. 다른 위치의 원천 세포주 8개 이상을 가진 목표는 1군이었다. 따라서 기존 다세포주 모델의 성능은 같은 원천 구성을 유지한 채 위치만 바꾸어 재현할 수 없었다. U2OS의 80군은 222개의 원천 plate별 짝을 갖는다. 이 비교는 **평가 웰과 인덱스가 같아야만 약물 신호가 나오는 것은 아님**을 보인다. U2OS 하나를 쓴 성능이 낮은 원인을 위치, 세포주 구성, 원천 표본 수 각각으로 나누지는 못했다. [위치·약물 짝 채점](data/jhh7_position_attribution/score/matched.tsv)

## 기존 다세포주 예측의 사후 재채점

1 µM 101군의 농도·fold·plate·DMSO 역할 평균을 제거한 뒤, 원 배율의 RMSE는 같은 입력 State보다 5.13% [99% 구간 3.83, 6.57], 같은 pool의 다른 약물보다 4.31% [2.90, 5.82] 낮았다. 같은 풀에서 반응 방향으로 약물을 맞힌 Top-1은 32.0%, State는 8.9%, 우연 기준은 3.0%였다. 이 재채점은 pool 평균을 제거하며 물리 RNA 웰 인덱스를 모두 통제하지는 않는다. [대비 점수](data/jhh7_position_attribution/score/centered.tsv) · [검색 지원과 구간](data/jhh7_position_attribution/score/retrieval.tsv)

원 절대 RMSE는 같은 입력 State 대비 101군 모두에서 낮았고, 같은 pool 다른 약물 대비 87군, B0 대비 68군에서 낮았다. 각 비교에서 제곱오차 절감이 가장 큰 20군을 제거하면 점추정 RMSE 개선은 각각 4.01%, 1.46%, 0.40%였다. 97개 scaffold 재표집에서는 원 비교의 99% 구간이 세 비교 모두에서 개선 방향이었다. 0.1 µM에서는 상위 20군 제거 뒤 B0 대비 RMSE 이득이 사라졌다. [화합물 기여](data/jhh7_position_attribution/score/influence.tsv) · [Scaffold 재표집](data/jhh7_position_attribution/score/scaffold.tsv)

명세 `4d01f3a`의 예측 `31704` → 동결 `31705` → 별도 채점 `31706`을 실행했다. 원 절대 점수와 99% 구간은 동결 원표와 일치했다. 예측과 채점 해시를 검증했다. 대상은 원 JHH-7 단일 plate에 한정했고 R18 예약 발현에는 접근하지 않았다. [작업 원장](data/jhh7_position_attribution/jobs.tsv) · [동결](data/jhh7_position_attribution/freeze.json)

## CMap HUH7에서 별도 평가 지원

HUH7 24 h의 121개 실측 농도를 메타데이터로 점검했다. 주요 두 농도의 고유 QC 지원은 2.5 µM 49군·91웰과 10 µM 41군·72웰이었다. 세 RNA plate의 DMSO 39웰은 R18 제외 뒤 모두 차단됐다. GEO 별칭까지 제외하면 두 주요 농도에 남는 처리 웰도 0개였다. 모든 농도에서 세 역할 DMSO 조건을 충족한 평가 웰은 0개였다. 감사 작업 `31703`은 종료 코드 0이었으며 발현 행렬은 열지 않았다. [농도별 지원](data/jhh7_position_attribution/huh7-support/support.tsv) · [Plate별 DMSO](data/jhh7_position_attribution/huh7-support/plates.tsv) · [메타데이터 영수증](data/jhh7_position_attribution/huh7-support/receipt.json)
