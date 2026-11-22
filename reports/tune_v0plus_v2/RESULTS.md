# V0+ v2 간암 약물 감수성

## 내부 네 라운드

GDSC2·CTRPv2·PRISM20Q2의 65,685 관측과 991 계통을 원래 3-fold 계통 분리로 유지했다. 바깥 개발 fold 내에서 약물별 모델, 잔차 배율, 표현 및 메타 가중치를 정했다. 표의 지표는 소스와 폴드의 동등 평균이다. Precision@5는 예측 상위 5개와 실측 상위 5개의 평균 교집합 비율이며 동률은 기대 회수율로 반영했다.

| 개발 설정 | 약물 내 Spearman | 약물 내 Pearson | 세포 내 NDCG@5 | 세포 내 Precision@5 | 세포 내 Spearman |
|---|---:|---:|---:|---:|---:|
| 기존 약물별 Ridge | .3660 | .4040 | .8930 | .6270 | .7340 |
| 기존 CellHit | .3560 | .3910 | .8910 | .6240 | .7330 |
| 기존 V0+ v1 | .3500 | .3970 | .8890 | .6190 | .7320 |
| R1 약물별 RBF | .3757 | .4111 | .8930 | .6228 | .7334 |
| R2 약물별 XGBoost | .3689 | .4079 | .8942 | .6273 | .7345 |
| R2 적응형 잔차 축소 | .3723 | .4084 | .8943 | .6269 | .7335 |
| R3 CMap+FM two-tower | .3181 | .3506 | .8865 | .6171 | .7155 |
| R3 SmoothL1 two-tower | .3211 | .3504 | .8887 | .6220 | .7202 |
| R4 Pairwise rank | **.3902** | **.4287** | **.9008** | .6340 | **.7458** |
| R4 Pairwise top-5 | .3890 | .4281 | .8995 | **.6365** | .7445 |

- R1은 여러 발현 정규화, HVG 256–2,048, 약물 특이 상관 유전자·공개 표적, Ridge, ElasticNet, PLS, RBF 및 cosine 커널을 각 약물의 inner fold에서 비교했다. RBF의 평균 약물 내 Spearman과 Pearson은 .3757과 .4111이었다. 세포별 약물 순위 지표의 개선은 제한적이었다.
- R2는 약물마다 별도 LightGBM, CatBoost, XGBoost를 학습했다. source-aligned 약물 anchor에서 예측 residual을 분리했고, 약물별 배율은 OOF 신호와 전체 shrinkage prior로 정했다.
- R3은 저랭크 bilinear, two-tower 및 hypernetwork에 RNA, scGPT·Geneformer, MoLFormer·CLAMP를 결합했다. CMap 2020의 비간 111개 세포 맥락에서 QC를 통과한 10,999개 24시간 서명으로 68개 중 65개 약물의 보조 표현을 구성했다. 이중 축 손실, SmoothL1 단독, 간·고형암 도메인 가중을 비교했다.
- R4는 비음수 스태킹, 약물별 편차 축소, native 어세이의 세포 내 약물쌍 손실로 학습한 anchor 보정을 비교했다. 설정 선택에는 LIMORE 반응의 개별 행을 쓰지 않았다.

[개발 지표](evidence/cv_macro.tsv) · [소스·fold별 지표](evidence/cv_by_source_fold.tsv) · [TKI 및 표적군 에러 진단](evidence/target_failure_diagnostics.tsv) · [CMap 특징 영수증](evidence/cmap_receipt.json) · [FM 및 간 홀드아웃](evidence/neural_fm_receipt.json)

## 동결 후 LIMORE 단일 평가

성인 HCC 50개 독립 계통, 68개 약물, 3,400개 Activity Area 관측에서 모든 모델을 같은 행에 채점했다. 개발 결과로 두 후보를 봉인했고 Pairwise rank를 기본 모델로 지정했다.

| 방법 | 약물 내 Spearman | 약물 내 Pearson | 세포 내 NDCG@5 | 세포 내 Precision@5 | 세포 내 Spearman |
|---|---:|---:|---:|---:|---:|
| 약물 평균 | .0000 | .0000 | .7606 | .2920 | .5313 |
| 약물별 Ridge | **.2399** | .2678 | .7484 | .2880 | .5168 |
| CellHit | .1013 | .1503 | .7738 | .2960 | .5574 |
| V0+ v1 | .1537 | .1907 | .7558 | .3080 | .5482 |
| Ridge + V0+ v1 | .2335 | .2748 | .7519 | .3000 | .5435 |
| **v2 Pairwise rank** | .2252 | .2748 | .7911 | **.3120** | **.5600** |
| v2 Pairwise top-5 | .2246 | **.2767** | **.7927** | .3040 | .5571 |

| 기본 v2 − 비교군 | 약물 내 Spearman [paired 95%] | 약물 내 Pearson [paired 95%] | NDCG@5 [paired 95%] | Precision@5 [paired 95%] | 세포 내 Spearman [paired 95%] |
|---|---:|---:|---:|---:|---:|
| 약물별 Ridge | −.0147 [−.0745, .0420] | +.0070 [−.0485, .0540] | +.0427 [.0293, .0564] | +.0240 [.0000, .0480] | +.0432 [.0328, .0535] |
| CellHit | +.1238 [.0507, .1892] | +.1246 [.0570, .1922] | +.0173 [.0012, .0339] | +.0160 [.0000, .0360] | +.0025 [−.0053, .0103] |
| V0+ v1 | +.0715 [−.0298, .1705] | +.0842 [−.0218, .1877] | +.0353 [.0154, .0563] | +.0040 [−.0240, .0280] | +.0118 [.0011, .0222] |

약물 내 95% 구간은 계통과 약물을 함께 짝 재표집했고, 세포 내 구간은 계통을 재표집했다. LIMORE의 실측 반응은 앞선 프로젝트에서 사용된 이력이 있다. 이 실행에서는 모델·하이퍼파라미터·스태킹을 먼저 고정하고 등록된 LIMORE 결과를 한 번만 채점했다.

HCC TKI 4종의 약물 내 Spearman / Pearson은 기본 v2 .1692 / .3298, Ridge .0842 / .2625, CellHit .1920 / .3078, V0+ v1 .0672 / .2120이었다. Sorafenib의 Spearman은 기본 v2 .1995, Ridge .0605, CellHit .1349, V0+ v1 −.0763이었다. TKI별 성적과 EGFR·FGFR·mTOR·HDAC 하위 분석은 채점 원표에 있다.


| LIMORE TKI | 기본 v2 Spearman / Pearson | 약물별 Ridge | CellHit | V0+ v1 |
|---|---:|---:|---:|---:|
| Sorafenib | .1995 / .3737 | .0605 / .1926 | .1349 / .3378 | −.0763 / .0536 |
| Cabozantinib | −.0652 / .0857 | −.1367 / .0297 | .1556 / .2040 | −.0561 / .0436 |
| Regorafenib | .3416 / .4585 | .2226 / .3450 | .3634 / .4458 | .2814 / .3310 |
| Lenvatinib | .2010 / .4014 | .1903 / .4828 | .1143 / .2436 | .1196 / .4196 |

**승격 판정.** 내부 다섯 지표와 LIMORE 세포 내 순위는 개선되었다. LIMORE에서 Ridge의 약물 내 Spearman이 기본 v2보다 .0147 높아 두 축 동시 기준에는 미달했다. 기본 v2는 연구용 약물 순위 우선 후보이며, 전체 개인화 성능의 동시 승격은 보류한다.

[동결 영수증](evidence/limore_forecast_freeze.json) · [LIMORE 원표 및 두 후보의 paired 구간](evidence/limore_score.json) · [CLI 재생](evidence/offline_receipt.json)

## 오가노이드·PDX·환자 감사

| 감사 | 기본 v2 | V0+ v1 | 판단 |
|---|---:|---:|---|
| Broutier 2개 HCC 오가노이드, 7개 약물 | NDCG@5 .7996, 세포 내 Spearman .5357 | .8596, .5536 | 작은 기술 평가에서 v1 우세 |
| GSE90653 16 PDX와 Sorafenib | Spearman −.4559 [95% −.8063, .0506] | −.3235 [−.7798, .2489] | v1 대비 −.1324 [paired 95% −.8842, .6256] |
| TCGA-LIHC 368명, 68개 약물 | v1 순위와 평균 Spearman .9075 | 기준 | 임상 약물 반응 관측 없음 |

TCGA에서 순도·면역·기질 보정 후 FDR 5% 미만 약물 수를 비교했다. CTNNB1-only 대 TP53-only는 v2 33/68, v1 34/68이었다. Hoshida S1/S2 대 S3는 둘 다 46/68이었다. 약물별 median absolute partial rho는 순도·면역·기질 순으로 v2 .072/.221/.185, v1 .081/.112/.144였다. 이 값은 환자 전사체에 따른 **모델 점수** 차이를 나타낸다. 기본 v2에서 면역과 기질의 연관이 더 컸으며 환자 감수성 향상으로 해석할 근거는 없다.

오프라인 재생은 207개 등록 요청 중 204개를 유한한 값으로 출력했고, 등록 패널 밖 3개는 명시적으로 반환했다. FM 없이 CPU에서 실행했다.

[3개 감사 영수증](evidence/audit_score.json) · [TCGA 아형·변이 표](evidence/tcga_driver_subtype.tsv) · [TCGA 순도 표](evidence/tcga_purity.tsv) · [TCGA 경로 표](evidence/tcga_panels.tsv) · [작업 원장](evidence/jobs.tsv) · [모델 카드](model_card.md)
