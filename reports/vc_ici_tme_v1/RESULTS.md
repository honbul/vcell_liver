# Track 7 — HCC anti-PD-1 external benchmark

## GSE202069

| model | participants | responders | auroc | auprc | balanced_accuracy |
| --- | --- | --- | --- | --- | --- |
| ayers_gep18 | 17 | 8 | 0.694 (0.375–0.972) | 0.814 (0.605–0.975) | 0.500 (0.500–0.500) |
| cd274 | 17 | 8 | 0.653 (0.361–0.917) | 0.695 (0.483–0.946) | 0.500 (0.500–0.500) |
| elasticnet | 17 | 8 | 0.403 (0.139–0.708) | 0.507 (0.366–0.747) | 0.451 (0.278–0.632) |
| fraction_path_ridge | 17 | 8 | 0.125 (0.000–0.347) | 0.337 (0.328–0.460) | 0.375 (0.250–0.500) |
| ifng | 17 | 8 | 0.639 (0.319–0.917) | 0.757 (0.557–0.942) | 0.500 (0.500–0.500) |
| ifng6 | 17 | 8 | 0.653 (0.333–0.931) | 0.769 (0.562–0.950) | 0.500 (0.500–0.500) |
| lightgbm | 17 | 8 | 0.222 (0.028–0.486) | 0.362 (0.333–0.506) | 0.333 (0.167–0.500) |
| mean | 17 | 8 | 0.500 (0.500–0.500) | 0.471 (0.471–0.471) | 0.500 (0.500–0.500) |
| nnls_purity | 17 | 8 | 0.903 (0.667–1.000) | 0.942 (0.812–1.000) | 0.688 (0.500–0.875) |
| ridge | 17 | 8 | 0.444 (0.152–0.736) | 0.590 (0.382–0.814) | 0.368 (0.181–0.556) |
| tide_author | 17 | 8 | 0.681 (0.375–0.931) | 0.771 (0.566–0.946) | 0.500 (0.500–0.500) |
| tme_gnn | 17 | 8 | 0.292 (0.056–0.569) | 0.391 (0.343–0.613) | 0.236 (0.056–0.472) |
| tme_no_edges | 17 | 8 | 0.264 (0.028–0.542) | 0.459 (0.339–0.679) | 0.375 (0.188–0.500) |
| tme_no_fractions | 17 | 8 | 0.236 (0.000–0.514) | 0.459 (0.335–0.686) | 0.396 (0.222–0.576) |
| tme_no_pathways | 17 | 8 | 0.139 (0.000–0.361) | 0.340 (0.327–0.452) | 0.236 (0.056–0.465) |

## GSE140901

| model | participants | responders | auroc | auprc | balanced_accuracy |
| --- | --- | --- | --- | --- | --- |
| cd274 | 24 | 6 | 0.426 (0.157–0.722) | 0.367 (0.196–0.669) | 0.500 (0.500–0.500) |
| elasticnet | 24 | 6 | 0.500 (0.259–0.759) | 0.272 (0.216–0.505) | 0.417 (0.250–0.611) |
| fraction_path_ridge | 24 | 6 | 0.630 (0.370–0.852) | 0.442 (0.249–0.743) | 0.639 (0.417–0.861) |
| ifng | 24 | 6 | 0.185 (0.028–0.426) | 0.181 (0.172–0.306) | 0.500 (0.500–0.500) |
| ifng6 | 24 | 6 | 0.648 (0.380–0.889) | 0.422 (0.262–0.806) | 0.528 (0.500–0.583) |
| lightgbm | 24 | 6 | 0.620 (0.389–0.824) | 0.330 (0.251–0.642) | 0.472 (0.306–0.667) |
| mean | 24 | 6 | 0.500 (0.500–0.500) | 0.250 (0.250–0.250) | 0.500 (0.500–0.500) |
| nnls_purity | 24 | 6 | 0.500 (0.213–0.796) | 0.318 (0.214–0.679) | 0.500 (0.500–0.500) |
| ridge | 24 | 6 | 0.481 (0.231–0.722) | 0.274 (0.212–0.563) | 0.500 (0.500–0.500) |
| tme_gnn | 24 | 6 | 0.546 (0.315–0.778) | 0.288 (0.226–0.535) | 0.500 (0.500–0.500) |
| tme_no_edges | 24 | 6 | 0.630 (0.324–0.889) | 0.478 (0.253–0.801) | 0.500 (0.500–0.500) |
| tme_no_fractions | 24 | 6 | 0.667 (0.435–0.880) | 0.386 (0.268–0.736) | 0.500 (0.500–0.500) |
| tme_no_pathways | 24 | 6 | 0.269 (0.074–0.481) | 0.196 (0.179–0.302) | 0.528 (0.500–0.583) |

## Paired primary-cohort differences

Differences compare the TME model to each baseline and ablation on the same held patients. The column bounds are percentile bootstrap intervals over patients.

| comparison | auroc | auroc_low | auroc_high | auprc | auprc_low | auprc_high |
| --- | --- | --- | --- | --- | --- | --- |
| ayers_gep18 | -0.4028 | -0.7778 | 0.0000 | -0.4224 | -0.5796 | -0.1331 |
| cd274 | -0.3611 | -0.7083 | 0.0278 | -0.3037 | -0.5280 | -0.0503 |
| elasticnet | -0.1111 | -0.3750 | 0.1250 | -0.1156 | -0.2499 | 0.0488 |
| fraction_path_ridge | 0.1667 | 0.0000 | 0.3472 | 0.0537 | -0.0037 | 0.1958 |
| ifng | -0.3472 | -0.7361 | 0.0694 | -0.3660 | -0.5377 | -0.0589 |
| ifng6 | -0.3611 | -0.7361 | 0.0278 | -0.3778 | -0.5453 | -0.0807 |
| lightgbm | 0.0694 | -0.2361 | 0.3611 | 0.0292 | -0.0802 | 0.1939 |
| mean | -0.2083 | -0.4444 | 0.0694 | -0.0795 | -0.1277 | 0.1426 |
| nnls_purity | -0.6111 | -0.9306 | -0.1944 | -0.5505 | -0.6517 | -0.2571 |
| ridge | -0.1528 | -0.4028 | 0.0833 | -0.1989 | -0.3446 | 0.0074 |
| tide_author | -0.3889 | -0.7222 | -0.0556 | -0.3802 | -0.5394 | -0.0924 |
| tme_no_edges | 0.0278 | -0.1667 | 0.2361 | -0.0681 | -0.1969 | 0.1046 |
| tme_no_fractions | 0.0556 | -0.1667 | 0.3056 | -0.0678 | -0.2109 | 0.1056 |
| tme_no_pathways | 0.1528 | -0.1111 | 0.4306 | 0.0511 | -0.0464 | 0.2393 |

## Simulated TME attenuation

The table ranks hypothetical graph interventions by mean probability change among measured nonresponders. The score uses frozen predictions, and the labels define the subgroup at scoring. Changes quantify graph sensitivity. Finn et al. tested atezolizumab plus bevacizumab in a distinct randomized population. [IMbrave150](https://doi.org/10.1056/NEJMoa1915745)

| axis | participants | change in percentage points | 95% lower bound | 95% upper bound |
| --- | --- | --- | --- | --- |
| TGFB1 | 9 | +0.00237 | +0.00098 | +0.00365 |
| VEGFA_KDR | 9 | +0.00234 | +0.00081 | +0.00406 |
| TIM3 | 9 | +0.00114 | -0.00077 | +0.00309 |
| SPP1_CD44 | 9 | +0.00055 | -0.00081 | +0.00234 |
| LAG3 | 9 | +0.00015 | -0.00279 | +0.00275 |
| WNT | 9 | -0.00033 | -0.00074 | +0.00012 |
| TIGIT | 9 | -0.00553 | -0.00964 | -0.00169 |

## Exploratory HCC expression

GSE141198 supplies 148 RNA profiles. The pooled fraction quantiles appear in the companion table. A matched objective anti-PD-1 response endpoint is unavailable for this cohort.

## TCGA-LIHC exploratory immune subtypes

Primary-tumor STAR expression was verified for 371 donors. The author subtype manifest matched 362 by exact sample barcode. These distributions do not measure ICI response. [PanImmune study](https://doi.org/10.1016/j.immuni.2018.03.023)

| subtype | patients | feature | median | q25 | q75 |
| --- | --- | --- | --- | --- | --- |
| C1 wound healing | 22 | malignant | 0.8801 | 0.7583 | 0.9117 |
| C1 wound healing | 22 | cd8 | 0.0000 | 0.0000 | 0.0000 |
| C1 wound healing | 22 | ifng6 | -0.3445 | -0.5414 | -0.1971 |
| C2 IFN-gamma dominant | 45 | malignant | 0.7704 | 0.6580 | 0.8747 |
| C2 IFN-gamma dominant | 45 | cd8 | 0.0000 | 0.0000 | 0.0101 |
| C2 IFN-gamma dominant | 45 | ifng6 | 0.2887 | 0.1384 | 0.5856 |
| C3 inflammatory | 135 | malignant | 0.9215 | 0.8657 | 0.9593 |
| C3 inflammatory | 135 | cd8 | 0.0000 | 0.0000 | 0.0000 |
| C3 inflammatory | 135 | ifng6 | -0.0878 | -0.2237 | 0.0430 |
| C4 lymphocyte depleted | 159 | malignant | 0.9629 | 0.9329 | 0.9759 |
| C4 lymphocyte depleted | 159 | cd8 | 0.0000 | 0.0000 | 0.0000 |
| C4 lymphocyte depleted | 159 | ifng6 | -0.2098 | -0.3436 | -0.0188 |
| C6 TGF-beta dominant | 1 | malignant | 0.9554 | 0.9554 | 0.9554 |
| C6 TGF-beta dominant | 1 | cd8 | 0.0005 | 0.0005 | 0.0005 |
| C6 TGF-beta dominant | 1 | ifng6 | 0.5870 | 0.5870 | 0.5870 |
