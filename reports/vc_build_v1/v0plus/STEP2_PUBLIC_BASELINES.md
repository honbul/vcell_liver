# V0+ 2단계 강한 비교군

후속 감사에서 이 결과의 CTRPv2 학습 코어에 단일 약물 두 개로 잘못 등록된 혼합 처리 1,394행이 확인됐다. 정정 코어와 원 자료를 별도로 고정했다. [자료 오류와 정정](SOURCE_PANEL_EXTENSION.md)

GDSC2·CTRPv2·PRISM20Q2의 65,685개 native AUC 관측을 991개 계통으로 묶어 nested CV했다. 표는 세 소스 각각에서 계산한 바깥쪽 보류 점수의 동등 평균이다. 얕은 모델과 공개 비교군 모두 보정한 같은 약물 anchor를 받았다. 외부 RPKM·AA 정답은 이 설정 선택에 사용하지 않았다.

| 방법 | 약물 내 Spearman | 약물 내 Pearson | NDCG@5 | Precision@5 | 세포 내 Spearman |
|---|---:|---:|---:|---:|---:|
| 약물 평균 | .000 | .000 | .873 | .589 | .688 |
| 약물+세포 가산 | .175 | .200 | .873 | .589 | .688 |
| 약물별 Ridge | .366 | .404 | .893 | .627 | .734 |
| 약물별 ElasticNet | .332 | .374 | .891 | .624 | .730 |
| LightGBM | .327 | .366 | .890 | .622 | .730 |
| CatBoost GPU | .314 | .353 | .884 | .613 | .724 |
| CellHit | .356 | .391 | .891 | .624 | .733 |
| Velodrome inductive | .351 | .389 | .891 | .624 | .726 |
| Velodrome transductive | .333 | .368 | .887 | .616 | .725 |
| CODE-AE inductive | .326 | .363 | .887 | .617 | .725 |
| CODE-AE transductive | .326 | .363 | .888 | .616 | .725 |

최선 전통 비교군은 약물별 Ridge이고 최선 공개 inductive 비교군은 CellHit다. 최종 학습 설정은 전체 개발 fold의 inner 결과로 정했다. Ridge는 alpha 1000·고변이 유전자 512개와 표적/서명 입력, ElasticNet은 alpha 0.02, LightGBM·CatBoost는 depth 7·650 trees, CellHit는 depth 5의 2개 저자 ensemble 멤버를 골랐다. 가산 모델의 잔차 배율은 0.25다. 가산 모델은 세포마다 전체 약물에 같은 값을 더하므로 그 세포의 약물 순서는 약물 평균과 동일하다.

ElasticNet의 일부 약한 벌점 설정에서 수렴 경고가 있었다. 등록된 선택 과정과 결과는 보존했다. CellHit·Velodrome·CODE-AE는 공개 구현을 연속 AUC로 적응해 재적합했다. Transductive 두 방법은 보류 집합 RNA를 정렬 단계에만 썼다. LIMORE 50개 결과는 4단계 최종 동결 채점에 함께 보고한다. [설정 및 소스별 원표](final_settings.json) · [모델 명세](MODEL_REGISTRATION.md) · [작업 원장](RUN_LEDGER.tsv)
