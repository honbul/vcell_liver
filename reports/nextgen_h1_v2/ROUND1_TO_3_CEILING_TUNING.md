# H1-v2 inner ceiling search

The signed inner searches evaluated 3,313 fold-aggregated recipes. Each recipe was compared on three held inner folds per task. Supervised anchor search cross-fitted inside each of those fit folds.

## Round 1 anchor ordering

| Policy | Mode | Mean ΔNDCG@5 vs K=0 | Mean Δ drug rho vs K=0 | Residual R² | Mean diversity | Selected tasks |
|---|---|---:|---:|---:|---:|---:|
| RNA-Residual-Greedy | anchor | +0.0029 | +0.4362 | 0.2481 | 0.9170 | 1 |
| RNA-Residual-Greedy | dual | +0.0006 | +0.2125 | 0.1861 | 0.9170 | 3 |
| RNA-Residual-LogDet | anchor | +0.0018 | +0.4275 | 0.2118 | 0.9157 | 0 |
| RNA-Residual-LogDet | dual | -0.0019 | +0.2064 | 0.1453 | 0.9157 | 0 |
| Supervised-Greedy-Dual | anchor | +0.0086 | +0.4536 | 0.2605 | 0.9167 | 7 |
| Supervised-Greedy-Dual | dual | +0.0027 | +0.2216 | 0.1888 | 0.9164 | 8 |
| Supervised-Greedy-NDCG | anchor | +0.0063 | +0.4465 | 0.2532 | 0.9168 | 4 |
| Supervised-Greedy-NDCG | dual | +0.0034 | +0.2137 | 0.1826 | 0.9170 | 2 |
| chemistry | anchor | +0.0095 | +0.4356 | 0.2512 | 0.9251 | 2 |
| chemistry | dual | +0.0052 | +0.2044 | 0.1835 | 0.9247 | 1 |
| logdet | anchor | +0.0034 | +0.4295 | 0.2076 | 0.9167 | 2 |
| logdet | dual | -0.0014 | +0.2008 | 0.1409 | 0.9159 | 1 |
| random-971 | anchor | +0.0100 | +0.4297 | 0.2404 | 0.9121 | 0 |
| random-971 | dual | +0.0031 | +0.2003 | 0.1689 | 0.9116 | 0 |
| random-972 | anchor | -0.0023 | +0.4179 | 0.1910 | 0.9126 | 0 |
| random-972 | dual | -0.0042 | +0.1926 | 0.1266 | 0.9118 | 0 |
| random-973 | anchor | +0.0080 | +0.4447 | 0.2558 | 0.9157 | 0 |
| random-973 | dual | +0.0029 | +0.2130 | 0.1901 | 0.9156 | 0 |
| random-3seed mean | anchor | 0.0053 | 0.4308 | — | — | 0 |
| random-3seed mean | dual | 0.0006 | 0.2020 | — | — | 0 |

The policy lock used the non-random methods. The three random seeds estimated the performance of random ordering. The random mean exceeded the locked policy utility in 2 task–modality combinations (LICOB-2 dual (+0.0015), LIMORE-2 anchor (+0.0016)).

The primary comparison preserves the complete-case denominator and one fit-reserved Q8 per inner split. The exploratory LICOB 16-candidate reserve holds a common Q fixed across all three supervised policies. The anchor table includes mechanism counts and per-fold diagnostics. [Round 1 table](H1V2_INNER_ANCHORS.tsv)

## Round 2 and Round 3 family comparators

| Study | Modality | Family | Mean R2 utility | Mean R3 utility | Mean NDCG@5 | Mean drug rho | Numerical boundary trials pending |
|---|---|---|---:|---:|---:|---:|---:|
| CTRPv2 | anchor | A | 0.9762 | 0.9763 | 0.9430 | 0.5558 | 0 |
| CTRPv2 | anchor | B | 0.9740 | 0.9744 | 0.9418 | 0.5424 | 0 |
| CTRPv2 | anchor | C | 0.9687 | 0.9695 | 0.9377 | 0.5301 | 0 |
| CTRPv2 | anchor | D | 0.9759 | 0.9759 | 0.9428 | 0.5508 | 0 |
| CTRPv2 | dual | A | 0.9870 | 0.9872 | 0.9509 | 0.6058 | 0 |
| CTRPv2 | dual | B | 0.9829 | 0.9829 | 0.9479 | 0.5832 | 0 |
| CTRPv2 | dual | C | 0.9814 | 0.9815 | 0.9466 | 0.5804 | 0 |
| CTRPv2 | dual | D | 0.9843 | 0.9846 | 0.9489 | 0.5959 | 0 |
| GDSC2 | anchor | A | 0.9768 | 0.9771 | 0.9541 | 0.3824 | 0 |
| GDSC2 | anchor | B | 0.9772 | 0.9775 | 0.9568 | 0.3445 | 0 |
| GDSC2 | anchor | C | 0.9731 | 0.9734 | 0.9538 | 0.3255 | 0 |
| GDSC2 | anchor | D | 0.9789 | 0.9794 | 0.9569 | 0.3753 | 0 |
| GDSC2 | dual | A | 0.9835 | 0.9836 | 0.9571 | 0.4422 | 0 |
| GDSC2 | dual | B | 0.9857 | 0.9859 | 0.9596 | 0.4372 | 0 |
| GDSC2 | dual | C | 0.9824 | 0.9828 | 0.9580 | 0.4130 | 0 |
| GDSC2 | dual | D | 0.9848 | 0.9852 | 0.9591 | 0.4351 | 0 |
| LICOB | anchor | A | 0.9136 | 0.9148 | 0.8710 | 0.3650 | 0 |
| LICOB | anchor | B | 0.9106 | 0.9106 | 0.8661 | 0.3704 | 0 |
| LICOB | anchor | C | 0.9054 | 0.9067 | 0.8655 | 0.3430 | 0 |
| LICOB | anchor | D | 0.9116 | 0.9116 | 0.8664 | 0.3761 | 0 |
| LICOB | dual | A | 0.9095 | 0.9111 | 0.8684 | 0.3559 | 0 |
| LICOB | dual | B | 0.8937 | 0.8937 | 0.8608 | 0.2744 | 0 |
| LICOB | dual | C | 0.9067 | 0.9067 | 0.8648 | 0.3490 | 0 |
| LICOB | dual | D | 0.9060 | 0.9060 | 0.8626 | 0.3622 | 0 |
| LIMORE | anchor | A | 0.9290 | 0.9297 | 0.9030 | 0.4459 | 0 |
| LIMORE | anchor | B | 0.9284 | 0.9288 | 0.9016 | 0.4531 | 0 |
| LIMORE | anchor | C | 0.9238 | 0.9289 | 0.9004 | 0.4751 | 0 |
| LIMORE | anchor | D | 0.9320 | 0.9323 | 0.9042 | 0.4677 | 0 |
| LIMORE | dual | A | 0.9357 | 0.9366 | 0.9061 | 0.5095 | 0 |
| LIMORE | dual | B | 0.9301 | 0.9301 | 0.9001 | 0.5002 | 0 |
| LIMORE | dual | C | 0.9304 | 0.9353 | 0.9059 | 0.4897 | 0 |
| LIMORE | dual | D | 0.9360 | 0.9360 | 0.9052 | 0.5132 | 0 |
| LIMORE_to_LICOB | anchor | A | 0.8945 | 0.8955 | 0.8607 | 0.5787 | 0 |
| LIMORE_to_LICOB | anchor | B | 0.8846 | 0.8897 | 0.8567 | 0.5496 | 0 |
| LIMORE_to_LICOB | anchor | C | 0.8651 | 0.8710 | 0.8378 | 0.5541 | 0 |
| LIMORE_to_LICOB | anchor | D | 0.9033 | 0.9033 | 0.8689 | 0.5733 | 0 |
| PRISM20Q2 | anchor | A | 0.9874 | 0.9874 | 0.9609 | 0.4427 | 0 |
| PRISM20Q2 | anchor | B | 0.9849 | 0.9852 | 0.9600 | 0.4202 | 0 |
| PRISM20Q2 | anchor | C | 0.9812 | 0.9837 | 0.9589 | 0.4128 | 0 |
| PRISM20Q2 | anchor | D | 0.9869 | 0.9870 | 0.9608 | 0.4379 | 0 |
| PRISM20Q2 | dual | A | 0.9948 | 0.9950 | 0.9654 | 0.4942 | 0 |
| PRISM20Q2 | dual | B | 0.9932 | 0.9939 | 0.9651 | 0.4802 | 0 |
| PRISM20Q2 | dual | C | 0.9900 | 0.9910 | 0.9627 | 0.4708 | 0 |
| PRISM20Q2 | dual | D | 0.9943 | 0.9950 | 0.9655 | 0.4915 | 0 |

The fitted policy was shared across families within each modality. Selection used NDCG@5 plus 0.06 × within-drug Spearman, with 0.12 × Spearman for LICOB.

- **A** varied EB priors, signal shrinkage and regularization. Its product kernel fits measured drug–sample pairs in the tensor product of the RNA/anchor and structure/target feature spaces.
- **B** compared L2, Huber, LambdaRank@5 and regression/rank blends. The development group was the LightGBM rank group.
- **C** trained self-attention over the anchor set, query cross-attention and gated RNA FiLM. SmoothL1, Pearson and ListNet or top-five pairwise loss were applied to the native ordering.
- **D** used an inner cross-fit EB prediction as a feature in the LambdaRank shortlist reranker.

### Boundary expansion and diagnostics

Each numerical expansion is tied to an incumbent, a previous range and the additional tested edge. Natural caps include unit-interval mixture weights, valid attention head divisors and positive penalties. Remaining numerical proposals, if any, appear in the table above. An empirical turnover applies to the tested configurations and split support.

[Grid scores](H1V2_INNER_GRID.tsv) · [Boundary trace](H1V2_INNER_BOUNDARIES.tsv) · [Selected recipes](H1V2_INNER_WINNERS.tsv) · [Train and validation curves](H1V2_LEARNING_CURVES.tsv) · [Tissue, mechanism and TKI errors](H1V2_INNER_ERRORS.tsv) · [Budget and noise sensitivity](H1V2_INNER_BUDGET.tsv)

| Family | Boundary recipes | Targeted refinements | Refined recipe wins | Median winning utility gain over Round 2 |
|---|---:|---:|---:|---:|
| A | 22 | 31 | 17 | +0.0001 |
| B | 76 | 31 | 4 | +0.0000 |
| C | 78 | 31 | 2 | +0.0007 |
| D | 71 | 31 | 4 | +0.0000 |

### Matched numerical turnover

| Family | Parameter | Lower probes | Median lower Δ utility | Upper probes | Median upper Δ utility | Two-sided brackets |
|---|---|---:|---:|---:|---:|---:|
| A | alpha | 2 | -0.0045 | 4 | -0.0044 | 2 |
| A | ridge | 2 | -0.0001 | 2 | -0.0005 | 2 |
| A | rna_weight | 0 | — | 1 | -0.0063 | 0 |
| A | width | 6 | -0.0009 | 6 | -0.0006 | 6 |
| B | blend | 3 | -0.0056 | 3 | -0.0044 | 3 |
| B | children | 3 | -0.0105 | 3 | -0.0062 | 3 |
| B | cols | 5 | -0.0062 | 3 | -0.0180 | 3 |
| B | depth | 2 | -0.0064 | 2 | -0.0035 | 2 |
| B | l2 | 4 | -0.0052 | 4 | -0.0033 | 4 |
| B | leaves | 1 | -0.0060 | 1 | -0.0055 | 1 |
| B | lr | 5 | -0.0004 | 5 | -0.0039 | 5 |
| B | rounds | 5 | -0.0009 | 5 | +0.0000 | 5 |
| B | subsample | 4 | -0.0008 | 3 | -0.0012 | 3 |
| C | decay | 1 | +0.0000 | 0 | — | 0 |
| C | dropout | 1 | -0.0001 | 0 | — | 0 |
| C | epochs | 1 | -0.0021 | 1 | -0.0001 | 1 |
| C | hidden | 15 | -0.0023 | 18 | -0.0009 | 15 |
| C | lr | 6 | -0.0035 | 6 | -0.0035 | 6 |
| C | rank_weight | 3 | -0.0020 | 5 | -0.0030 | 3 |
| D | blend | 1 | -0.0024 | 1 | -0.0048 | 1 |
| D | cols | 4 | -0.0021 | 1 | -0.0007 | 1 |
| D | depth | 2 | -0.0046 | 2 | -0.0079 | 2 |
| D | l2 | 2 | -0.0019 | 2 | -0.0060 | 2 |
| D | leaves | 1 | -0.0066 | 1 | -0.0117 | 1 |
| D | rounds | 1 | -0.0010 | 5 | +0.0000 | 1 |
| D | stage.alpha | 23 | -0.0035 | 4 | -0.0004 | 4 |
| D | stage.ridge | 9 | -0.0012 | 9 | -0.0008 | 9 |
| D | top | 1 | -0.0010 | 1 | -0.0004 | 1 |

Each probe fixes all other recipe fields and uses the nearest tested value on its side. Each reported median uses the probe-minus-winner utility differences for that parameter and direction. One-sided rows show the side tested. The complete grid also records joint recipe changes. [Grid scores](H1V2_INNER_GRID.tsv) · [Boundary trace](H1V2_INNER_BOUNDARIES.tsv)

| Family | Parameter at its tested limit | Value | Selected fold-modality recipes |
|---|---|---:|---:|
| A | alpha | 0 | 3 |
| A | rna_weight | 0 | 1 |
| B | cols | 1 | 2 |
| B | subsample | 1 | 1 |
| C | rank_weight | 0 | 5 |
| D | blend | 1 | 1 |
| D | cols | 1 | 3 |

### Selected train and validation curves

| Study | Mode | Family | Head / metric | Validation first | Validation best | Train at best | Validation last | Mean best step |
|---|---|---|---|---:|---:|---:|---:|---:|
| CTRPv2 | anchor | B | rank / ndcg@5 | 0.8632 | 0.9369 | 0.9435 | 0.9358 | 68.0 |
| CTRPv2 | anchor | B | reg / rmse | 2.1454 | 1.7318 | 1.7263 | 1.7325 | 130.4 |
| CTRPv2 | anchor | C | set / NDCG@5 | 0.9349 | 0.9377 | 0.9380 | 0.9367 | 52.8 |
| CTRPv2 | anchor | D | rank / ndcg@5 | 0.8524 | 0.9385 | 0.9409 | 0.9377 | 38.6 |
| CTRPv2 | dual | B | rank / ndcg@5 | 0.8510 | 0.9424 | 0.9531 | 0.9419 | 63.5 |
| CTRPv2 | dual | B | reg / rmse | 1.8440 | 1.6501 | 1.5720 | 1.6504 | 73.3 |
| CTRPv2 | dual | C | set / NDCG@5 | 0.9452 | 0.9467 | 0.9446 | 0.9443 | 20.0 |
| CTRPv2 | dual | D | rank / ndcg@5 | 0.8552 | 0.9421 | 0.9460 | 0.9416 | 43.4 |
| GDSC2 | anchor | B | rank / ndcg@5 | 0.9116 | 0.9570 | 0.9567 | 0.9565 | 29.6 |
| GDSC2 | anchor | B | reg / rmse | 0.0877 | 0.0799 | 0.0773 | 0.0799 | 76.6 |
| GDSC2 | anchor | C | set / NDCG@5 | 0.9532 | 0.9541 | 0.9533 | 0.9485 | 45.6 |
| GDSC2 | anchor | D | rank / ndcg@5 | 0.8967 | 0.9572 | 0.9566 | 0.9570 | 29.3 |
| GDSC2 | dual | B | rank / ndcg@5 | 0.9148 | 0.9596 | 0.9625 | 0.9585 | 25.8 |
| GDSC2 | dual | B | reg / rmse | 0.0781 | 0.0737 | 0.0677 | 0.0739 | 66.8 |
| GDSC2 | dual | C | set / NDCG@5 | 0.9568 | 0.9582 | 0.9562 | 0.9505 | 27.8 |
| GDSC2 | dual | D | rank / ndcg@5 | 0.8632 | 0.9599 | 0.9591 | 0.9594 | 21.4 |
| LICOB | anchor | B | rank / ndcg@5 | 0.6905 | 0.8523 | 0.8900 | 0.8418 | 18.7 |
| LICOB | anchor | B | reg / rmse | 0.2004 | 0.1848 | 0.1578 | 0.1854 | 54.6 |
| LICOB | anchor | C | set / NDCG@5 | 0.8472 | 0.8657 | 0.8757 | 0.8435 | 30.0 |
| LICOB | anchor | D | rank / ndcg@5 | 0.6948 | 0.8620 | 0.8909 | 0.8520 | 24.6 |
| LICOB | dual | B | rank / ndcg@5 | 0.7151 | 0.8555 | 0.9050 | 0.8457 | 21.2 |
| LICOB | dual | B | reg / rmse | 0.2021 | 0.1916 | 0.1485 | 0.1926 | 39.3 |
| LICOB | dual | C | set / NDCG@5 | 0.8522 | 0.8652 | 0.8389 | 0.8460 | 28.9 |
| LICOB | dual | D | rank / ndcg@5 | 0.7165 | 0.8488 | 0.9081 | 0.8377 | 21.8 |
| LIMORE | anchor | B | rank / ndcg@5 | 0.7646 | 0.8954 | 0.9152 | 0.8889 | 15.0 |
| LIMORE | anchor | B | reg / rmse | 0.1680 | 0.1451 | 0.1222 | 0.1458 | 51.4 |
| LIMORE | anchor | C | set / NDCG@5 | 0.8835 | 0.9004 | 0.9102 | 0.8844 | 42.8 |
| LIMORE | anchor | D | rank / ndcg@5 | 0.7823 | 0.8976 | 0.9199 | 0.8882 | 12.0 |
| LIMORE | dual | B | rank / ndcg@5 | 0.7586 | 0.8834 | 0.9252 | 0.8767 | 24.0 |
| LIMORE | dual | B | reg / rmse | 0.1564 | 0.1380 | 0.1075 | 0.1384 | 67.3 |
| LIMORE | dual | C | set / NDCG@5 | 0.8893 | 0.9061 | 0.8995 | 0.8981 | 44.4 |
| LIMORE | dual | D | rank / ndcg@5 | 0.7999 | 0.8963 | 0.9242 | 0.8891 | 11.6 |
| LIMORE_to_LICOB | anchor | B | rank / ndcg@5 | 0.7241 | 0.8591 | 0.8951 | 0.8538 | 27.7 |
| LIMORE_to_LICOB | anchor | B | reg / rmse | 0.1701 | 0.1411 | 0.1245 | 0.1413 | 53.3 |
| LIMORE_to_LICOB | anchor | C | set / NDCG@5 | 0.8093 | 0.8378 | 0.8628 | 0.8247 | 50.0 |
| LIMORE_to_LICOB | anchor | D | rank / ndcg@5 | 0.7580 | 0.8829 | 0.9044 | 0.8673 | 17.7 |
| PRISM20Q2 | anchor | B | rank / ndcg@5 | 0.8740 | 0.9551 | 0.9567 | 0.9547 | 62.5 |
| PRISM20Q2 | anchor | B | reg / rmse | 0.0943 | 0.0821 | 0.0809 | 0.0821 | 102.0 |
| PRISM20Q2 | anchor | C | set / NDCG@5 | 0.9540 | 0.9589 | 0.9571 | 0.9549 | 52.8 |
| PRISM20Q2 | anchor | D | rank / ndcg@5 | 0.9141 | 0.9541 | 0.9557 | 0.9535 | 35.6 |
| PRISM20Q2 | dual | B | rank / ndcg@5 | 0.9263 | 0.9576 | 0.9612 | 0.9571 | 29.3 |
| PRISM20Q2 | dual | B | reg / rmse | 0.0894 | 0.0782 | 0.0749 | 0.0783 | 121.6 |
| PRISM20Q2 | dual | C | set / NDCG@5 | 0.9604 | 0.9629 | 0.9616 | 0.9595 | 23.3 |
| PRISM20Q2 | dual | D | rank / ndcg@5 | 0.9257 | 0.9602 | 0.9610 | 0.9598 | 39.3 |

For each fold, the table reports the first, best and final recorded validation points and the train value at the best validation point. A late validation decline despite continued fit separates overfitting from lack of early learning. Neural checkpoint selection used the combined ranking and Spearman objective. The full selected traces include neural within-drug Spearman. [Learning curves](H1V2_LEARNING_CURVES.tsv)

### Subgroup errors in the two leading Round 2 families

| Study | Modality | Group type | Group | Summed validation entries | Mean fold MAE |
|---|---|---|---|---:|---:|
| CTRPv2 | anchor | drug | Regorafenib | 2404 | 1.0785 |
| CTRPv2 | anchor | drug | Cabozantinib | 2382 | 0.9780 |
| CTRPv2 | anchor | drug | Sorafenib | 2176 | 0.9501 |
| CTRPv2 | anchor | drug | Lenvatinib | 2480 | 0.9399 |
| CTRPv2 | anchor | mechanism | DHFR | 2320 | 2.3873 |
| CTRPv2 | anchor | mechanism | GPX4 | 2468 | 2.3379 |
| CTRPv2 | anchor | tissue | myeloid | 7648 | 1.5192 |
| CTRPv2 | anchor | tissue | soft tissue | 2006 | 1.4121 |
| CTRPv2 | dual | drug | Regorafenib | 2404 | 1.0890 |
| CTRPv2 | dual | drug | Cabozantinib | 2382 | 0.9818 |
| CTRPv2 | dual | drug | Sorafenib | 2176 | 0.9801 |
| CTRPv2 | dual | drug | Lenvatinib | 2480 | 0.9457 |
| CTRPv2 | dual | mechanism | DHFR | 2320 | 2.2382 |
| CTRPv2 | dual | mechanism | NAMPT | 4642 | 2.1191 |
| CTRPv2 | dual | tissue | myeloid | 7648 | 1.4106 |
| CTRPv2 | dual | tissue | peripheral nervous system | 2052 | 1.3131 |
| GDSC2 | anchor | drug | Sorafenib | 2468 | 0.0435 |
| GDSC2 | anchor | mechanism | Survivin | 2740 | 0.1895 |
| GDSC2 | anchor | mechanism | MET, KDR, FLT3 | 2712 | 0.1460 |
| GDSC2 | anchor | tissue | myeloid | 6464 | 0.0813 |
| GDSC2 | anchor | tissue | lymphoid | 18968 | 0.0701 |
| GDSC2 | dual | drug | Sorafenib | 2468 | 0.0419 |
| GDSC2 | dual | mechanism | Survivin | 2740 | 0.1360 |
| GDSC2 | dual | mechanism | DHFR | 2556 | 0.1352 |
| GDSC2 | dual | tissue | myeloid | 6464 | 0.0758 |
| GDSC2 | dual | tissue | lymphoid | 18968 | 0.0642 |
| LICOB | anchor | drug | Sorafenib | 152 | 0.1436 |
| LICOB | anchor | drug | Regorafenib | 152 | 0.1360 |
| LICOB | anchor | drug | Lenvatinib | 152 | 0.0951 |
| LICOB | anchor | mechanism | NAMPT Metabolism Metabolism | 152 | 0.3309 |
| LICOB | anchor | mechanism | Survivin Apoptosis Apoptosis | 152 | 0.2610 |
| LICOB | anchor | tissue | Liver | 10336 | 0.1382 |
| LICOB | dual | drug | Sorafenib | 152 | 0.1492 |
| LICOB | dual | drug | Regorafenib | 152 | 0.1404 |
| LICOB | dual | drug | Lenvatinib | 152 | 0.1022 |
| LICOB | dual | mechanism | NAMPT Metabolism Metabolism | 152 | 0.3491 |
| LICOB | dual | mechanism | Survivin Apoptosis Apoptosis | 152 | 0.2676 |
| LICOB | dual | tissue | Liver | 10336 | 0.1434 |
| LIMORE | anchor | drug | Cabozantinib | 200 | 0.0715 |
| LIMORE | anchor | drug | Regorafenib | 200 | 0.0447 |
| LIMORE | anchor | drug | Sorafenib | 200 | 0.0398 |
| LIMORE | anchor | drug | Lenvatinib | 200 | 0.0395 |
| LIMORE | anchor | mechanism | NAMPT | 200 | 0.2811 |
| LIMORE | anchor | mechanism | Survivin | 200 | 0.2600 |
| LIMORE | anchor | tissue | liver | 12000 | 0.0986 |
| LIMORE | dual | drug | Cabozantinib | 200 | 0.0714 |
| LIMORE | dual | drug | Regorafenib | 200 | 0.0487 |
| LIMORE | dual | drug | Sorafenib | 200 | 0.0456 |
| LIMORE | dual | drug | Lenvatinib | 200 | 0.0406 |
| LIMORE | dual | mechanism | NAMPT | 200 | 0.2619 |
| LIMORE | dual | mechanism | Survivin | 200 | 0.2477 |
| LIMORE | dual | tissue | liver | 12000 | 0.0950 |
| LIMORE_to_LICOB | anchor | drug | Regorafenib | 100 | 0.0358 |
| LIMORE_to_LICOB | anchor | drug | Lenvatinib | 100 | 0.0343 |
| LIMORE_to_LICOB | anchor | mechanism | NAMPT | 100 | 0.2501 |
| LIMORE_to_LICOB | anchor | mechanism | Survivin | 100 | 0.2273 |
| LIMORE_to_LICOB | anchor | tissue | liver | 4500 | 0.0937 |
| PRISM20Q2 | anchor | drug | Lenvatinib | 1490 | 0.0473 |
| PRISM20Q2 | anchor | mechanism | MAP2K1, MAP2K2 | 1548 | 0.1247 |
| PRISM20Q2 | anchor | mechanism | EGFR, ERBB2, ERBB4 | 2976 | 0.1156 |
| PRISM20Q2 | anchor | tissue | bowel | 5130 | 0.0686 |
| PRISM20Q2 | anchor | tissue | pleura | 1192 | 0.0679 |
| PRISM20Q2 | dual | drug | Lenvatinib | 1490 | 0.0465 |
| PRISM20Q2 | dual | mechanism | MAP2K1, MAP2K2 | 1548 | 0.1101 |
| PRISM20Q2 | dual | mechanism | PARP | 876 | 0.1061 |
| PRISM20Q2 | dual | tissue | bowel | 5130 | 0.0671 |
| PRISM20Q2 | dual | tissue | liver | 1742 | 0.0663 |

The counts sum across family and fold evaluations. Each outcome can contribute to both families. The table lists the two highest-error supported tissue and mechanism groups per study and modality, plus all observed HCC TKIs. MAE uses the study-specific response scale.

### LICOB reference sample size

| Modality | Fraction | Mean fit donors | Mean NDCG@5 | Mean within-drug Spearman |
|---|---:|---:|---:|---:|
| anchor | 0.5 | 8.8 | 0.8432 | 0.3153 |
| anchor | 0.75 | 12.9 | 0.8526 | 0.3348 |
| anchor | 1 | 16.9 | 0.8710 | 0.3720 |
| dual | 0.5 | 8.8 | 0.8178 | 0.3009 |
| dual | 0.75 | 12.9 | 0.8498 | 0.3471 |
| dual | 1 | 16.9 | 0.8667 | 0.3612 |

The two leading families retained their fixed inner recipes, anchor ordering and validation donors while reference fit sizes varied. This estimates sample-size sensitivity conditional on the chosen strategy. [Sample-size results](H1V2_LICOB_SAMPLE_SIZE.tsv)

- **anchor** mean fit size increased from 8.8 to 16.9 donors. The corresponding mean changes were +0.0278 NDCG@5 and +0.0567 within-drug Spearman.
- **dual** mean fit size increased from 8.8 to 16.9 donors. The corresponding mean changes were +0.0489 NDCG@5 and +0.0602 within-drug Spearman.
