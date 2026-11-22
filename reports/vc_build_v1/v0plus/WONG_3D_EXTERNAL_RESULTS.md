# Wong HCC 오가노이드 독립 평가

후속 감사에서 이 결과의 CTRPv2 학습 코어에 단일 약물 두 개로 잘못 등록된 혼합 처리 1,394행이 확인됐다. 표와 동결 영수증은 당시 이력이다. 정정 단독 처리 코어를 적용한 성능은 별도 검증 대상이다. [자료 오류와 정정](SOURCE_PANEL_EXTENSION.md)

**환자별 약물 전체 순위는 개선됐다. 약물 내 환자 감수성 구별과 NDCG@5도 점추정은 양수이며, 두 개선의 95% 구간은 0을 포함한다.**

GDSC2·CTRPv2·PRISM만으로 적합한 7개 방법을 독립 성인 HCC 오가노이드 10개 × 27약물에 적용했다. S6 기저 TPM과 S7 mean inhibition AUC에서 270개 등록 행 모두 관측이 있었다. 소스 RNA와 공통인 유전자는 17,741개다. 100개 중 90개 약물의 구조를 Broad 전체 물리 표본 ID로, 나머지 10개를 PubChem CAS로 해소했다. 기존 학습 패널과 전체 parent InChIKey가 일치한 27개는 모두 Broad에서 확인됐다. [사전 명세](WONG_3D_PROTOCOL.md) · [코호트 등록 영수증](wong_registration_receipt.json) · [원논문](https://doi.org/10.1016/j.jhepr.2025.101426)

| 방법 | 약물 내 Spearman | 약물 내 Pearson | 오가노이드 내 NDCG@5 | Precision@5 | 오가노이드 내 Spearman |
|---|---:|---:|---:|---:|---:|
| 약물 평균 | .000 | .000 | .761 | .340 | .451 |
| 약물별 Ridge | .092 | .094 | .758 | .420 | .511 |
| CellHit | −.001 | −.005 | .727 | .320 | .464 |
| Velodrome inductive | −.024 | −.016 | .751 | .360 | .464 |
| CODE-AE inductive | .021 | .080 | .721 | .300 | .443 |
| V0+ inductive | .061 | .060 | .780 | .380 | .495 |
| Ridge+V0+ 고정 평균 | .072 | .097 | .755 | .340 | .503 |

V0+의 오가노이드 내 전체 순위 Spearman은 약물 평균보다 +.044 [95% .021, .070], CellHit보다 +.032 [.014, .049] 높았다. 약물 평균 대비 NDCG@5 차이는 +.019 [−.006, .049]였다. 약물 내 Spearman과 Pearson은 약물 평균보다 각각 +.061 [−.160, .291], +.060 [−.159, .280]이었다. 약물 내 비교는 환자와 약물을 함께 재표집했고 오가노이드 내 비교는 환자를 짝지어 재표집했다.

V0+의 NDCG@5는 CellHit보다 +.053 [.022, .083] 높았다. CellHit 대비 약물 내 Spearman 차이는 +.062 [−.185, .315]였다. Ridge는 V0+보다 약물 내 Spearman, Precision@5, 오가노이드 내 Spearman에서 높은 점수를 기록했다. V0+ 대비 Ridge의 약물 내 Spearman 차이 +.031의 95% 구간은 [−.211, .275]다.

## 약물군과 실패 모드

| 층 | 약물 수 | V0+ 약물 내 Spearman | Ridge | CellHit |
|---|---:|---:|---:|---:|
| HCC TKI | 4 | −.018 | .048 | −.042 |
| 비특이 세포독성 | 9 | .084 | .162 | .216 |
| 주석상 키나아제 억제제 | 5 | .108 | .173 | .064 |
| HDAC | 2 | −.388 | −.418 | −.424 |

V0+의 개별 HCC TKI Spearman은 sorafenib −.515, regorafenib −.248, cabozantinib .370, lenvatinib .321이었다. Sorafenib은 LIMORE에서 −.076, Hu PDX 단일약물 기술 평가에서 −.324로 방향이 일치했다. 서로 다른 표본과 측정 종말점에서 같은 약물의 전이가 어려웠다는 관측이다. 계통 수가 10개라 약물별 결과는 기술 통계다. [LIMORE 성적](STEP4_EXTERNAL_RESULTS.md) · [PDX 및 공개 자료 감사](STEP4_3D_PUBLIC_ACCESS_FOLLOWUP.md)

등록 `32700`, 팔별 예측 `32701–32705`, 병합·동결 `32706`, 별도 채점 `32707`. 총 1,890개 예측의 동결 SHA-256은 `ac128ab49120ab4ec694f0180cf1bf935acb38aba079d16656b1f947a98897d0`이다. [동결 영수증](wong_forecast_receipt.json) · [모든 지표·약물별 원표](wong_score.json)
