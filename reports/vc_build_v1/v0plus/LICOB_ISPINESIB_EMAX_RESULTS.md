# Ispinesib 최고 농도 정렬과 LICOB 외부 성적

**LICOB 38명의 최고 농도 Emax에서는 공동 기준에 미달했다.** 농도를 맞춘 모델의 약물 내 Spearman은 .033이었고 같은 소스 세포의 발행 AUC와 원자료 AUC 대조는 각각 .126과 .153이었다. [사전 명세](LICOB_ISPINESIB_EMAX_PROTOCOL.md)

## 물리 농도와 2D 반복

복원된 7약물의 PRISM 최고 농도는 10–10.6681 µM에 있었다. 6약물의 LICOB 최고 농도는 20 µM였다. Ispinesib만 LICOB의 최고 두 농도 3.333과 10 µM를 PRISM에서 덮었다. PRISM의 가장 낮은 측정은 .000610 µM이고 LICOB 설계의 가장 낮은 농도는 약 .000508 µM여서, 가장 낮은 농도까지 포함한 AUC 정렬은 이 약물에서도 불가능했다. [PRISM 원자료 농도](prism_20q2_raw_dose_receipt.json) · [LICOB 공개 S5](https://www.ebi.ac.uk/biostudies/studies/S-EPMC10949980)

세 모델 모두 571개 PRISM 세포의 기저 RNA를 사용했다. 그중 성인 HCC는 13개였다. 558개 동일 계통의 PRISM 검출 플레이트 반복에서 원 범위 AUC Spearman은 .859 [95% .830, .883], 최고 두 농도 Emax는 .874 [.845, .898]였다. 2D에서 기술 반복이 높은 표적을 사용해도 외부 전이가 개선되는지 직접 시험했다. [소스 영수증](licob_ispinesib_emax_forecast_receipt.json)

## 동결 외부 결과

| 학습 기준 | LICOB Emax Spearman | LICOB Emax Pearson | 보조 LICOB AUC Spearman |
|---|---:|---:|---:|
| 약물 평균 | .000 | .000 | .000 |
| 발행 적합 AUC | .126 | −.045 | −.160 |
| 실측 원범위 AUC | .153 | −.037 | −.140 |
| **농도 일치 Emax** | **.033** | **−.033** | **−.168** |

농도 일치 모형의 Emax Spearman 자체의 95% 구간은 [−.318, +.392]였다. 발행 AUC 대비 짝지은 차이는 −.093 [−.255, +.049], 실측 원범위 AUC 대비 −.120 [−.294, +.033]였다. 이 수치는 사전 명세의 동일 38명·같은 약물 조건이다. 단일 화합물이므로 사람별 다약물 선별과 임상 예측을 판정할 수 없다. [별도 채점 원표](licob_ispinesib_emax_score.json)

`72b2710`에서 명세를 고정했고 GPU `33736`은 43오가노이드의 172행을 동결했다. CPU `33737`은 공식 S8·S6을 별도로 읽어 채점했다. 두 작업은 종료 코드 0이었다. 동결 SHA-256 `935720b8af4286791ca041048aa2b5050263a41eb51bb731b369ab2a5c4df826`, 채점 SHA-256 `44a7b0c2e616492ca4fe925741023819e3c9544e03d8f5ad2572a5c462b708e3`. [작업 원장](RUN_LEDGER.tsv)
