# H1 nested inner tuning

The 17 outer-fit tasks each used three independently hashed inner splits. The archived grids contain 35,736 configuration–fold evaluations across three rounds. Every line of the [inner family table](H1_INNER_FAMILY_WINNERS.tsv) represents one task, one round and one family winner at K=5 on the selected policy's shared Q. The G1 freeze stores the full per-split grid, including Round 3 K=0/3/8 and SHA-256 for every grid and model. The [freeze summary](inner_freeze_summary.json) includes selected penalties, scales, ranks and the complete split-level diagnostic values.

## Round 1

Each policy selected anchors from fit responses and produced its own held validation Q. Comparing a policy's raw NDCG to another policy's raw NDCG would also compare testing panels. Selection therefore used the paired NDCG difference against the best simple arm on that policy's Q. The three random seeds are controls. Each deterministic policy competed for adoption.

| Policy | Mean paired linear gain over best simple arm | Outer-fit tasks selected |
|---|---:|---:|
| logdet | +0.0029 | 6 |
| leverage | +0.0017 | 2 |
| chemistry | +0.0053 | 9 |
| random-971 | +0.0032 | 0 |
| random-972 | +0.0050 | 0 |
| random-973 | +0.0040 | 0 |
| random-3-seed-mean | +0.0041 | 0 |

There were 81–93 distinct Round 1 recipes per task, including the baselines. Sample covariance used pair-specific fit means and jointly observed responses. Ledoit–Wolf and OAS used a fit-centered working grid, with zero residual in an unobserved training cell. Scoring used observed values. The SVD sweep used fit-only PSD projection and rank truncation. See [all policy results](H1_ANCHOR_POLICIES.tsv).

| Study | Mean across inner folds of drug-specific correction SD | SD of sample-constant correction | Inner mean change after +2 SD in worst anchor | Tasks with 90th-percentile largest anchor z >3 |
|---|---:|---:|---:|---:|
| GDSC2 | 0.0250 | 0.0196 | 0.0145 | 0/3 |
| CTRPv2 | 0.6864 | 0.9750 | 0.5981 | 0/3 |
| PRISM20Q2 | 0.0188 | 0.0289 | 0.0190 | 0/3 |
| LIMORE | 0.0815 | 0.0737 | 0.0564 | 2/3 |
| LICOB | 0.0561 | 0.0405 | 0.0411 | 0/3 |
| LIMORE_to_LICOB | 0.0634 | 0.0816 | 0.0535 | 0/1 |
| LIMORE_to_Wong | 0.0597 | 0.0785 | 0.0438 | 1/1 |

The fitted correction varied across drugs in all studies. The two-SD stress probe perturbed the largest standardized anchor within each validation sample and re-predicted Q. High maximum anchor residuals triggered the broader Huber clipping grid in those tasks. The registered 2D and chemical priors tested whether local LICOB covariance could be stabilized at the small fit sample size. To keep native-scale errors comparable, the following comparisons are within each study.

| Study | Highest native-scale tissue MAE with ≥10 inner observations across folds | Largest mechanism MAE averaged over supported folds |
|---|---|---|
| GDSC2 | myeloid (0.0862) | nonselective_cytotoxics (0.1021) |
| CTRPv2 | myeloid (1.5333) | nonselective_cytotoxics (1.4575) |
| PRISM20Q2 | bowel (0.0698) | egfr (0.0706) |
| LIMORE | liver (0.0953) | hdac (0.1025) |
| LICOB | Liver (0.1410) | nonselective_cytotoxics (0.1688) |
| LIMORE_to_LICOB | liver (0.0965) | nonselective_cytotoxics (0.1075) |
| LIMORE_to_Wong | liver (0.0890) | hdac (0.1033) |

## Rounds 2 and 3

Round 2 compared 77–130 recipes per task. The EB comparison included the zero-shrinkage estimate and four nonzero mixture weights for each available prior. Huber applies a bounded score to the measured residual, winsorization uses the fit residual quantiles, and the rank transform uses fit-only empirical CDFs. Nonlinear ridge applies RBF or cosine similarity to anchor profiles. Bilinear ALS estimates factors from observed training entries with ridge stabilization. The RNA residual branch used predictions for each fit group from subfolds that omitted that group. The kernel branch compared RNA weights 0, 0.1, 0.25 and 0.5.

Round 3 began with the Round 1 linear winner and kept at most two Round 2 anchor-only families whose average NDCG gain over B4 reached 0.002, remained positive in at least two inner folds and preserved mean Precision@5 within 0.02. It tested half and double the prior ridge, then applied the same retention test again. B4 remained the fallback. Inner simple-arm selection included B0 through B4 where inputs were allowed. RNA entered the secondary procedure only on folds where its own paired gain over the final anchor-only recipe met the same criterion. The secondary procedure uses the anchor-only recipe on folds that missed that gate.

| Outer-fit task | Policy | Round 1 linear NDCG | Round 2 best anchor-only family and NDCG | Round 3 frozen family | R3 champion NDCG | R3 strongest simple NDCG | RNA gate |
|---|---|---:|---|---|---:|---:|---|
| CTRPv2-0 | chemistry | 0.9459 | oas 0.9459 | huber | 0.9460 | B1 0.9456 | Pass (+0.0052) |
| CTRPv2-1 | chemistry | 0.9470 | winsor 0.9471 | winsor | 0.9473 | B1 0.9467 | Pass (+0.0066) |
| CTRPv2-2 | leverage | 0.9532 | rank 0.9539 | rbf | 0.9540 | B1 0.9549 | Pass (+0.0059) |
| GDSC2-0 | logdet | 0.9369 | svd 0.9369 | svd | 0.9369 | B1 0.9367 | Pass (+0.0043) |
| GDSC2-1 | chemistry | 0.9537 | rbf 0.9551 | cosine | 0.9560 | B1 0.9519 | Miss (-0.0003) |
| GDSC2-2 | chemistry | 0.9528 | winsor 0.9529 | winsor | 0.9529 | B1 0.9544 | Pass (+0.0050) |
| LICOB-0 | logdet | 0.8880 | pan 0.8911 | chem | 0.8902 | B0 0.8798 | Miss (+0.0004) |
| LICOB-1 | logdet | 0.8964 | svd 0.8964 | svd | 0.8964 | B1 0.8840 | Miss (+0.0019) |
| LICOB-2 | chemistry | 0.8540 | svd 0.8540 | svd | 0.8540 | B0 0.8460 | Miss (-0.0009) |
| LIMORE-0 | logdet | 0.8818 | sample 0.8818 | sample | 0.8818 | B1 0.8670 | Miss (-0.0011) |
| LIMORE-1 | chemistry | 0.9176 | svd 0.9176 | svd | 0.9176 | B4 0.8820 | Miss (+0.0025) |
| LIMORE-2 | chemistry | 0.9121 | pan 0.9140 | pan | 0.9160 | B1 0.8984 | Miss (-0.0031) |
| LIMORE_to_LICOB-0 | chemistry | 0.8551 | pan 0.8559 | chem | 0.8557 | B4 0.8418 | Ineligible |
| LIMORE_to_Wong-0 | logdet | 0.8906 | rank 0.8927 | winsor | 0.8932 | B0 0.8797 | Ineligible |
| PRISM20Q2-0 | leverage | 0.9584 | winsor 0.9587 | winsor | 0.9587 | B1 0.9556 | Pass (+0.0059) |
| PRISM20Q2-1 | logdet | 0.9647 | chem 0.9648 | chem | 0.9648 | B1 0.9617 | Pass (+0.0033) |
| PRISM20Q2-2 | chemistry | 0.9674 | rank 0.9678 | rank | 0.9678 | B1 0.9615 | Pass (+0.0025) |

LICOB selected chemical EB in one fold and SVD covariance in two. The pan-cancer response prior was tested in all LICOB folds and did not make their final selection. LIMORE retained the pan-cancer prior in one fold. Nonlinear candidates cleared the anchor-only selection gate in two of the pan-cancer folds. ALS was tested but did not survive the final paired validation. All three HCC RNA gates missed in LICOB, and all three missed in LIMORE. Eight of nine pan-cancer folds qualified for the secondary RNA procedure. Exact winning configurations and the candidates advanced from Round 2 are recorded in the [freeze summary](inner_freeze_summary.json).

[동결 후 Inner 비교군 검사](H1_INNER_LOCKED_CONTROLS.tsv)는 각 task의 세 Inner validation fold에서 모든 K와 비교군을 재생했다. 같은 폴드의 A8·Q·평가 분모가 모든 팔에서 일치한다. N1은 앵커 패킷 전체를, N2는 RNA만 이동한다. 이 검사는 보류된 Outer 결과를 입력으로 사용하지 않았고 챔피언 선택을 바꾸지 않았다. [검사 영수증](inner_control_receipt.json)
