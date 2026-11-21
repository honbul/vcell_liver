# 동결 결과의 화합물·plate 기여도

**T4의 단순 배율 대비 이득과 R17 B의 donor 대비 이득은 상위 기여군과 개별 plate를 제외해도 남았다.** 원 동결과 처리·대조군의 물리 등록을 대조했고, 전체 지원의 원 점수와 paired 99% 구간을 재현했다. 아래 조정은 이미 채점한 결과의 사후 민감도다. [명세](ROBUSTNESS_PROTOCOL.md)

| 동결 비교 | RMSE가 감소한 parent | Parent·plate 재가중 99% 구간, 상대 RMSE | Parent·plate 재가중 99% 구간, pooled R² |
|---|---:|---:|---:|
| T4 시간 연산자 대 전역 배율 | 95 / 132 | +5.05~+9.75% | +5.04~+19.99%p |
| T4 시간 연산자 대 6 h 원 측정 | 123 / 132 | +9.76~+13.85% | +9.37~+27.63%p |
| T4 시간 연산자 대 별도 24 h 반복 | 83 / 132 | −2.52~+4.30% | −5.99~+8.28%p |
| R17 B 결합 대 동일약물 donor anchor | 398 / 493 | +1.59~+2.55% | +1.90~+4.57%p |

T4의 배율 대비 양의 RMSE 기여에서 상위 20군의 비중은 48.6%였고, 양의 SSE 기여에서는 51.9%였다. 각 지표에서 해당 상위 20군을 뺀 112군의 RMSE 이득은 3.95% 또는 4.73%, R² 이득은 9.06 또는 9.11%p였다. 12개 채점 plate를 하나씩 뺀 배율 대비 RMSE 이득은 7.03~8.14%였다. 별도 24 h 반복 대비의 우위는 두 재가중 구간이 모두 0을 포함했다. [T4 parent 기여](data/robustness/T4/parents.tsv) · [상위군 제외](data/robustness/T4/top_removed.tsv) · [plate 제외](data/robustness/T4/plate_removed.tsv)

R17 B 결합에서 상위 20군은 양의 RMSE 기여의 15.9%, 양의 SSE 기여의 26.9%였다. 각 지표의 상위 20군을 뺀 473군에서도 RMSE가 1.77% 또는 1.85% 낮았다. 41개 채점 plate를 하나씩 제외한 RMSE 감소는 1.96~2.15%였다. [B parent 기여](data/robustness/SL_B/parents.tsv) · [상위군 제외](data/robustness/SL_B/top_removed.tsv) · [plate 제외](data/robustness/SL_B/plate_removed.tsv)

양쪽 점수를 재계산한 작업 `31157`과 `31158`은 각 7초, CPU 2개, GPU 0개로 종료 코드 0이었다. 이 구간은 부모와 물리 plate가 교차한 자료에 대한 추가 점검이다. 독립 assay 또는 성인 HCC에서 얻는 확인 구간은 별도다. [T4 재가중 원표](data/robustness/T4/crossed.tsv) · [B 재가중 원표](data/robustness/SL_B/crossed.tsv) · [Slurm 원장](data/robustness/jobs.tsv)
