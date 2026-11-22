# 간암 2D 효능과 3D 현지 보정 확증

| 작업 완료 | 가설 통과 | 도구 완성 | 현장 효용 |
|---|---|---|---|
| Task 1–4 코드·동결·별도 채점·리포트·브랜치 푸시 | 2D 동시 승격 미통과, 3D 0/18 | 정정 구조 커널 CPU CLI 3,400행 재생 및 입력 거절 | 신규 실험 미실시. 공개 자료 후향 검증 |

**단일 2D 기준은 정정 구조 커널**이다. 사용 범위는 약물 내 감수성 비교용 연구 기준이다. 고정한 동시 승격 기준을 통과한 모델은 없었다. 세포 내 약물 순위와 약물 내 감수성 수치는 모델별로 해석한다. 현지 소수 라벨 보정의 주된 효용은 약물 기본 순위의 현지화였다. RNA 개인화의 추가 가치는 현재 공개 3D 코호트에서 미확증으로 판정했다. 운영 연구의 단순 기준으로 현지 K명 평균을 유지한다.

## Task 1 · 정정 65,678행 단일 원장

GDSC2·CTRPv2·PRISM20Q2의 1,014모델·991계통·68약물을 적합하고 독립 LIMORE 성인 HCC 50계통 × 68약물의 동일 Activity Area 3,400행을 한 번 채점했다. 명세가 열거한 9개 비교군을 각각 적합했다. 순위 이득 NDCG@5와 동률 기대 Precision@5를 모든 모델에 적용했다.

| 모델 | 약물 내 ρ | 약물 내 r | NDCG@5 | Precision@5 | 세포 내 ρ |
|---|---|---|---|---|---|
| 약물 평균 | 0.0000† | 0.0000† | 0.7102 | 0.2640 | 0.5304 |
| 약물 + 세포 가산 | 0.0623 | 0.0577 | 0.7102 | 0.2640 | 0.5304 |
| 약물별 Ridge | 0.2429 | 0.2716 | 0.7324 | 0.2760 | 0.5127 |
| CellHit 재적합 | 0.1040 | 0.1544 | 0.7485 | 0.2960 | 0.5535 |
| 정정 구조 커널 | 0.3363 | 0.3813 | 0.7397 | 0.2840 | 0.5461 |
| Ridge + 구조 커널 50:50 | 0.3067 | 0.3408 | 0.7356 | 0.2840 | 0.5373 |
| V0+ v2 Pairwise Rank | 0.2324 | 0.2800 | 0.7636 | 0.3000 | 0.5528 |
| V0+ v2 Pairwise Top-5 | 0.2326 | 0.2813 | 0.7682 | 0.3040 | 0.5493 |
| ECFP4 + 경로 ElasticNet | 0.3633 | 0.3668 | 0.7480 | 0.3400 | 0.5304 |

LIMORE, LICOB, Wong, Yang은 앞선 프로젝트에서 평가한 이력이 있다. 모든 수치는 고정된 비교 계약 하의 후향 통합 확증이다. † 약물 평균은 세포 사이에서 상수라 약물 내 ρ와 r이 수학적으로 미정의다. 대조 계산에는 0을 넣었고 TKI 원표의 해당 필드는 NA다.

학습표 SHA-256은 `1edd32ef38be60a742efe0dcc376e15910623c1da8cc27d923291bee8a4a38cc`이다. 진단 코드는 종전 LIMORE 평가식과 최대 절대 오차 1.1e-16였다. 2,400회 계통·약물 교차 재표집을 동일 예측에 적용했다. [단일 원장](UNIFIED_2D_LEDGER.tsv) · [paired 95% 구간](LIMORE_PAIRED_95CI.tsv) · [채점 영수증](score_receipt.json)

### 4개 HCC TKI 성적

| 모델 | 약물 | 계통 | Spearman ρ | Pearson r |
|---|---|---|---|---|
| 약물 평균 | Regorafenib | 50 | — | — |
| 약물 평균 | Sorafenib | 50 | — | — |
| 약물 평균 | Cabozantinib | 50 | — | — |
| 약물 평균 | Lenvatinib | 50 | — | — |
| 약물 + 세포 가산 | Regorafenib | 50 | 0.1882 | 0.1970 |
| 약물 + 세포 가산 | Sorafenib | 50 | 0.1954 | 0.2523 |
| 약물 + 세포 가산 | Cabozantinib | 50 | -0.1772 | -0.1939 |
| 약물 + 세포 가산 | Lenvatinib | 50 | 0.1736 | 0.2694 |
| 약물별 Ridge | Regorafenib | 50 | 0.2202 | 0.3447 |
| 약물별 Ridge | Sorafenib | 50 | 0.0622 | 0.1906 |
| 약물별 Ridge | Cabozantinib | 50 | -0.1367 | 0.0297 |
| 약물별 Ridge | Lenvatinib | 50 | 0.1958 | 0.4845 |
| CellHit 재적합 | Regorafenib | 50 | 0.3628 | 0.4449 |
| CellHit 재적합 | Sorafenib | 50 | 0.1146 | 0.3326 |
| CellHit 재적합 | Cabozantinib | 50 | 0.1556 | 0.2040 |
| CellHit 재적합 | Lenvatinib | 50 | 0.1141 | 0.2453 |
| 정정 구조 커널 | Regorafenib | 50 | 0.2486 | 0.4696 |
| 정정 구조 커널 | Sorafenib | 50 | 0.1750 | 0.4399 |
| 정정 구조 커널 | Cabozantinib | 50 | 0.1411 | 0.2009 |
| 정정 구조 커널 | Lenvatinib | 50 | 0.2952 | 0.4882 |
| Ridge + 구조 커널 50:50 | Regorafenib | 50 | 0.2535 | 0.4320 |
| Ridge + 구조 커널 50:50 | Sorafenib | 50 | 0.0856 | 0.3156 |
| Ridge + 구조 커널 50:50 | Cabozantinib | 50 | -0.0442 | 0.1167 |
| Ridge + 구조 커널 50:50 | Lenvatinib | 50 | 0.2404 | 0.5071 |
| V0+ v2 Pairwise Rank | Regorafenib | 50 | 0.4323 | 0.4576 |
| V0+ v2 Pairwise Rank | Sorafenib | 50 | 0.2018 | 0.3483 |
| V0+ v2 Pairwise Rank | Cabozantinib | 50 | -0.0228 | 0.2068 |
| V0+ v2 Pairwise Rank | Lenvatinib | 50 | 0.1794 | 0.3817 |
| V0+ v2 Pairwise Top-5 | Regorafenib | 50 | 0.4558 | 0.4695 |
| V0+ v2 Pairwise Top-5 | Sorafenib | 50 | 0.2207 | 0.3476 |
| V0+ v2 Pairwise Top-5 | Cabozantinib | 50 | -0.0074 | 0.1741 |
| V0+ v2 Pairwise Top-5 | Lenvatinib | 50 | 0.1762 | 0.3928 |
| ECFP4 + 경로 ElasticNet | Regorafenib | 50 | 0.0423 | 0.1517 |
| ECFP4 + 경로 ElasticNet | Sorafenib | 50 | -0.0358 | 0.0971 |
| ECFP4 + 경로 ElasticNet | Cabozantinib | 50 | 0.2927 | 0.2877 |
| ECFP4 + 경로 ElasticNet | Lenvatinib | 50 | 0.2471 | 0.2760 |

[TKI 원표](LIMORE_TKI.tsv)

## Task 2 · 2D 상호작용과 추천 다양성

| 모델 | 잔차 ρ [95%] | 고유 Top-5 / 50 | 쌍별 Jaccard |
|---|---|---|---|
| 약물 평균 | +0.0000 [+0.0000, +0.0000] | 1 | 1.0000 |
| 약물 + 세포 가산 | +0.0000 [+0.0000, +0.0000] | 1 | 1.0000 |
| 약물별 Ridge | +0.2477 [+0.1608, +0.3289] | 10 | 0.6823 |
| CellHit 재적합 | +0.1793 [+0.0904, +0.2683] | 7 | 0.8241 |
| 정정 구조 커널 | +0.2606 [+0.1540, +0.3557] | 6 | 0.7741 |
| V0+ v2 Pairwise Rank | +0.2952 [+0.1976, +0.3894] | 9 | 0.7195 |
| ECFP4 + 경로 ElasticNet | +0.0000 [+0.0000, +0.0000] | 1 | 1.0000 |

가산 모델은 계통별 높낮이를 바꿀 수 있지만 이중 중심화하면 상호작용이 0이다. 100회 독립 계통 derangement에서 RNA와 RNA 유도 벡터를 함께 옮겼다. [전체 개인화 감사](PERSONALIZATION_AUDIT.md)

## Task 3 · 3D 현지 라벨 보정

| 코호트 | K=4 모델 | NDCG@5 | 약물 내 ρ | 잔차 ρ |
|---|---|---|---|---|
| LICOB | c · EB 수축 | 0.7816 | 0.0000 | 0.0000 |
| LICOB | d · EB + 구조 커널 RNA | 0.7830 | 0.1012 | 0.1265 |
| LICOB | d · EB + Pairwise Rank RNA | 0.8042 | 0.1036 | 0.1576 |
| Wong | c · EB 수축 | 0.8588 | 0.0000 | 0.0000 |
| Wong | d · EB + 구조 커널 RNA | 0.8600 | -0.0109 | 0.0192 |
| Wong | d · EB + Pairwise Rank RNA | 0.8631 | 0.0561 | 0.0620 |
| Yang | c · EB 수축 | 0.8127 | 0.0000 | 0.0000 |
| Yang | d · EB + 구조 커널 RNA | 0.8057 | -0.0283 | -0.0507 |
| Yang | d · EB + Pairwise Rank RNA | 0.8151 | 0.0076 | 0.0017 |

[세 코호트의 K=2,4,8 전체 결과 및 환자 단위 전이표](FEWSHOT_3D_CONFIRM.md)

## Task 4 · 입력 검증과 논문 골격

정정 구조 커널 CPU 패키지가 3,400개 예측을 재생했다. 동결 대비 최대오차 2.22e-16, 단일 파일과 배치의 최대오차 1.86e-07, 단일 RNA·68약물 CPU CLI 중앙값 13.968초였다. 음수 입력, 전체 및 선택 유전자 미달, 모든 값 0, 미등록 exact parent를 검사했다. [패키지 영수증](package_replay_receipt.json) · [모델 카드](model_card.md) · [Paper 2 초안](PAPER2_OUTLINE_AND_TABLES.md)
