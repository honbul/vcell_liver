# T4 시간 연산자의 세포 맥락 전이 결과

**HEPG2에서 적합한 6→24 h 연산자를 일차 간세포에 적용하면 단순 배율보다 오차가 커졌다.** 같은 L1000에서 대상의 6 h 실측을 공통으로 제공하고, 다른 맥락에서 적합한 rank-16 연산자와 배율을 24 h 물리 RNA plate에서 평가했다. 역방향도 별도 기술 비교했다. [사전 명세](T4_CONTEXT_TRANSFER_PROTOCOL.md)

| 적합 → 평가 | 기준선 | 상대 RMSE 변화, 증가가 악화 [99% 구간] | pooled R² 차이, %p [99% 구간] |
|---|---|---:|---:|
| HEPG2 → PHH, 67군·105웰 | HEPG2 적합 배율 | +4.54% [3.21, 5.85] | −9.80 [−12.61, −7.17] |
| HEPG2 → PHH | PHH 적합 배율 | +6.18% [5.09, 7.33] | −12.11 [−14.80, −9.73] |
| PHH → HEPG2, 132군·268웰 | PHH 적합 배율 | +8.90% [7.90, 9.94] | −16.38 [−18.59, −14.24] |
| PHH → HEPG2 | HEPG2 적합 배율 | +9.65% [8.35, 10.97] | −19.00 [−21.38, −16.52] |

주 방향은 두 배율 대조에서 모두 사전 공동 기준을 통과하지 못했고 실용 이득도 구간상 배제됐다. HEPG2 연산자가 PHH의 보정 전 6 h 입력을 그대로 쓴 비교보다 RMSE를 3.53% 줄였으나 [1.24, 5.76], 전이 연산자의 pooled R²는 −8.39%였고 PHH 적합 배율은 +3.72%였다. PHH에서 같은 시간의 독립 plate 직접 복사 대비 두 차이 구간은 0을 포함했다. 역방향에서는 직접 복사보다 두 지표 모두 악화했다. [PHH 원표](data/t4_context_transfer/PHH/score/paired.tsv) · [HEPG2 원표](data/t4_context_transfer/HEPG2/score/paired.tsv)

적합에는 HEPG2에서 fold별 78–96군·53 plate, PHH에서 37–50군·21 plate를 썼다. 각 fold의 대상 parent 전체와 대상 plate를 다른 맥락 적합에서 제외했다. 순위와 ridge는 기존 DEV에서 사용한 16과 1로 고정했다. 두 맥락의 적합 약물·물리 plate 수가 다르므로 상호 전이 손실을 세포 생물학과 학습 지원의 차이로 나눌 수 없다. 결과는 이 두 개발 맥락의 저랭크 연산자에 해당한다. 성인 HCC의 독립 확인 자료와 R18 발현은 사용하지 않았다. [PHH 방향 적합 지원](data/t4_context_transfer/PHH/score/fitting.tsv) · [HEPG2 방향 적합 지원](data/t4_context_transfer/HEPG2/score/fitting.tsv)

명세 `8dbd19a`에서 G1 `t4-context.XlDSzix0`의 예측 `31474` → 동결 `31475` → 별도 채점 `31476`을 실행했다. 22개 작업이 모두 종료 코드 0이었고 GPU 사용은 0이다. 양쪽 동결·계산 코드·채점 표 해시를 확인했다. [작업 원장](data/t4_context_transfer/jobs.tsv) · [자원 내역](data/t4_context_transfer/resource_usage.tsv)
