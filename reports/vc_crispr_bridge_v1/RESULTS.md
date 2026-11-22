# Track 4 — HepG2 Perturb-seq and genetic-to-chemical transfer

Independent scorers verified frozen prediction manifests and held assay roles. Each panel is reported on the measured axis listed with its metrics.

## HepG2 transcriptional effects

| panel | model | measured_targets | centered_pearson | centered_rmse | top20_recall | top20_signed_recall |
| --- | --- | --- | --- | --- | --- | --- |
| gene_cold | conditional_mlp | 38 | 0.01119 | 0.11167 | 0.08026 | 0.07105 |
| gene_cold | elasticnet | 38 | 0.05956 | 0.11328 | 0.05526 | 0.04474 |
| gene_cold | lightgbm | 38 | 0.06409 | 0.11121 | 0.11842 | 0.10000 |
| gene_cold | mlp_esm2_only | 38 | 0.00542 | 0.11176 | 0.08026 | 0.07632 |
| gene_cold | mlp_no_context | 38 | 0.01605 | 0.11152 | 0.09079 | 0.08158 |
| gene_cold | mlp_string_only | 38 | 0.02264 | 0.11116 | 0.09211 | 0.08158 |
| gene_cold | no_change | 38 | 0.00000 | 0.11115 | 0.00000 | 0.00000 |
| gene_cold | ridge | 38 | 0.04043 | 0.11191 | 0.07368 | 0.06316 |
| gene_cold | same_gene_copy | 38 | -0.00210 | 0.11115 | 0.10921 | 0.10132 |
| gene_cold | source_mean | 38 | -0.00210 | 0.11115 | 0.10921 | 0.10132 |
| gene_cold | string_graph_gnn | 38 | 0.05837 | 0.11275 | 0.05921 | 0.04474 |
| gene_cold | target_expression_attenuation | 38 | 0.06091 | 0.11088 | 0.02237 | 0.02237 |
| same_gene | conditional_mlp | 79 | 0.02631 | 0.10651 | 0.08101 | 0.07532 |
| same_gene | elasticnet | 79 | 0.15211 | 0.10513 | 0.08544 | 0.07278 |
| same_gene | lightgbm | 79 | 0.21297 | 0.10420 | 0.15570 | 0.14620 |
| same_gene | mlp_esm2_only | 79 | 0.02144 | 0.10657 | 0.08671 | 0.07975 |
| same_gene | mlp_no_context | 79 | 0.03169 | 0.10632 | 0.09367 | 0.08671 |
| same_gene | mlp_string_only | 79 | 0.06123 | 0.10596 | 0.08924 | 0.08354 |
| same_gene | no_change | 79 | 0.00000 | 0.10612 | 0.00000 | 0.00000 |
| same_gene | ridge | 79 | 0.09781 | 0.10598 | 0.08861 | 0.08101 |
| same_gene | same_gene_copy | 79 | 0.21989 | 0.11989 | 0.16835 | 0.15949 |
| same_gene | source_mean | 79 | 0.00007 | 0.10612 | 0.10823 | 0.10190 |
| same_gene | string_graph_gnn | 79 | 0.14871 | 0.10514 | 0.06646 | 0.05949 |
| same_gene | target_expression_attenuation | 79 | 0.05180 | 0.10585 | 0.01266 | 0.01266 |

## DepMap Chronos

| panel | method | models | eligible_groups | within_group_spearman | rmse | mae |
| --- | --- | --- | --- | --- | --- | --- |
| gene_cold | conditional_mlp | 20 | 3569 | 0.00913 | 0.34827 | 0.19781 |
| gene_cold | elasticnet | 20 | 3569 | 0.00102 | 0.39098 | 0.24441 |
| gene_cold | lightgbm | 20 | 3569 | 0.00938 | 0.32392 | 0.19557 |
| gene_cold | mean | 20 | 3569 | 0.00000 | 0.43859 | 0.25191 |
| gene_cold | mlp_no_cell_context | 20 | 3569 | 0.00000 | 0.36030 | 0.20223 |
| gene_cold | mlp_no_esm2 | 20 | 3569 | 0.00766 | 0.32954 | 0.19292 |
| gene_cold | mlp_no_string | 20 | 3569 | 0.00518 | 0.37760 | 0.20977 |
| gene_cold | mlp_no_target_omics | 20 | 3569 | 0.00831 | 0.36022 | 0.20310 |
| gene_cold | ridge | 20 | 3569 | 0.00012 | 0.39089 | 0.24436 |
| gene_cold | source_gene_mean | 20 | 3569 | 0.00000 | 0.43859 | 0.25191 |
| gene_cold | target_omics_marker | 20 | 3569 | -0.00081 | 0.40222 | 0.24429 |
| same_gene | conditional_mlp | 20 | 10659 | 0.01404 | 0.27186 | 0.16185 |
| same_gene | elasticnet | 20 | 10659 | -0.00163 | 0.38134 | 0.24150 |
| same_gene | lightgbm | 20 | 10659 | 0.01185 | 0.22942 | 0.15406 |
| same_gene | mean | 20 | 10659 | 0.00000 | 0.42494 | 0.24627 |
| same_gene | mlp_no_cell_context | 20 | 10659 | 0.00000 | 0.30498 | 0.17594 |
| same_gene | mlp_no_esm2 | 20 | 10659 | 0.01007 | 0.30914 | 0.18244 |
| same_gene | mlp_no_string | 20 | 10659 | 0.00658 | 0.32311 | 0.18372 |
| same_gene | mlp_no_target_omics | 20 | 10659 | 0.00462 | 0.29346 | 0.17005 |
| same_gene | ridge | 20 | 10659 | -0.00094 | 0.38129 | 0.24145 |
| same_gene | source_gene_mean | 20 | 10659 | 0.00000 | 0.16539 | 0.11170 |
| same_gene | target_omics_marker | 20 | 10659 | 0.00074 | 0.39301 | 0.24265 |

## LIMORE drug and target holdout

Each fold selects chemistry-only, genetic-only and joint predictors using only its blocked validation. The contrasts below bootstrap paired test-drug Spearman deltas.

| split | contrast | test_drugs | delta_spearman | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| scaffold | joint_gain | 87 | 0.00085 | -0.01183 | 0.01303 |
| scaffold | genetic_only_gain | 87 | -0.00629 | -0.02441 | 0.01032 |
| unseen_target | joint_gain | 87 | 0.00857 | 0.00311 | 0.01475 |
| unseen_target | genetic_only_gain | 87 | -0.00402 | -0.01592 | 0.00688 |

The full fold-averaged comparison includes exact culture Chronos matches as a separate subset.

| split | support | method | folds | within_group_spearman | cell_ndcg5 | rmse |
| --- | --- | --- | --- | --- | --- | --- |
| scaffold | all | ABCB1 | 5 | 0.05107 | 0.41793 | 5.38262 |
| scaffold | all | ecfp4/elasticnet | 5 | 0.52729 | 0.36244 | 0.21383 |
| scaffold | all | ecfp4/lightgbm | 5 | 0.52350 | 0.38521 | 0.22311 |
| scaffold | all | ecfp4/mlp | 5 | 0.52421 | 0.45542 | 0.21014 |
| scaffold | all | ecfp4/ridge | 5 | 0.52852 | 0.44836 | 0.20183 |
| scaffold | all | genetic_observed/elasticnet | 5 | 0.52088 | 0.57760 | 0.19167 |
| scaffold | all | genetic_observed/lightgbm | 5 | 0.47871 | 0.48252 | 0.20842 |
| scaffold | all | genetic_observed/mlp | 5 | 0.51244 | 0.46777 | 0.20579 |
| scaffold | all | genetic_observed/ridge | 5 | 0.52669 | 0.55556 | 0.19301 |
| scaffold | all | joint_no_chronos/elasticnet | 5 | 0.52728 | 0.43345 | 0.24549 |
| scaffold | all | joint_no_chronos/lightgbm | 5 | 0.51978 | 0.43553 | 0.21150 |
| scaffold | all | joint_no_chronos/mlp | 5 | 0.50461 | 0.38660 | 0.22367 |
| scaffold | all | joint_no_chronos/ridge | 5 | 0.52848 | 0.43603 | 0.22808 |
| scaffold | all | joint_no_expression_signature/elasticnet | 5 | 0.52783 | 0.42008 | 0.23950 |
| scaffold | all | joint_no_expression_signature/lightgbm | 5 | 0.52866 | 0.40786 | 0.21616 |
| scaffold | all | joint_no_expression_signature/mlp | 5 | 0.50790 | 0.41143 | 0.22038 |
| scaffold | all | joint_no_expression_signature/ridge | 5 | 0.52840 | 0.41045 | 0.22595 |
| scaffold | all | joint_observed/elasticnet | 5 | 0.52800 | 0.42932 | 0.24628 |
| scaffold | all | joint_observed/lightgbm | 5 | 0.52496 | 0.42504 | 0.21458 |
| scaffold | all | joint_observed/mlp | 5 | 0.50296 | 0.41606 | 0.22044 |
| scaffold | all | joint_observed/ridge | 5 | 0.52860 | 0.43398 | 0.22829 |
| scaffold | all | joint_predicted/elasticnet | 5 | 0.52784 | 0.40981 | 0.24679 |
| scaffold | all | joint_predicted/lightgbm | 5 | 0.52535 | 0.41067 | 0.21436 |
| scaffold | all | joint_predicted/mlp | 5 | 0.50417 | 0.46489 | 0.21640 |
| scaffold | all | joint_predicted/ridge | 5 | 0.52858 | 0.46495 | 0.22838 |
| scaffold | all | max_loss | 5 | 0.04043 | 0.54550 | 1.11476 |
| scaffold | all | molformer/elasticnet | 5 | 0.52729 | 0.44733 | 0.23768 |
| scaffold | all | molformer/lightgbm | 5 | 0.51984 | 0.45628 | 0.21237 |
| scaffold | all | molformer/mlp | 5 | 0.51126 | 0.37573 | 0.22529 |
| scaffold | all | molformer/ridge | 5 | 0.52852 | 0.42648 | 0.22623 |
| scaffold | all | train_mean | 5 | 0.00000 | 0.41793 | 0.21389 |
| scaffold | measured_chronos | ABCB1 | 5 | 0.06402 | 0.41133 | 4.38279 |
| scaffold | measured_chronos | ecfp4/elasticnet | 5 | 0.51144 | 0.35049 | 0.21724 |
| scaffold | measured_chronos | ecfp4/lightgbm | 5 | 0.50732 | 0.36348 | 0.22756 |
| scaffold | measured_chronos | ecfp4/mlp | 5 | 0.50378 | 0.45372 | 0.21104 |
| scaffold | measured_chronos | ecfp4/ridge | 5 | 0.51167 | 0.43005 | 0.20503 |
| scaffold | measured_chronos | genetic_observed/elasticnet | 5 | 0.48551 | 0.56491 | 0.19483 |
| scaffold | measured_chronos | genetic_observed/lightgbm | 5 | 0.44374 | 0.50595 | 0.20460 |
| scaffold | measured_chronos | genetic_observed/mlp | 5 | 0.50004 | 0.48515 | 0.20802 |
| scaffold | measured_chronos | genetic_observed/ridge | 5 | 0.50763 | 0.54791 | 0.19560 |
| scaffold | measured_chronos | joint_no_chronos/elasticnet | 5 | 0.51113 | 0.42367 | 0.24724 |
| scaffold | measured_chronos | joint_no_chronos/lightgbm | 5 | 0.51043 | 0.42287 | 0.21571 |
| scaffold | measured_chronos | joint_no_chronos/mlp | 5 | 0.49488 | 0.38155 | 0.22513 |
| scaffold | measured_chronos | joint_no_chronos/ridge | 5 | 0.51204 | 0.40833 | 0.23123 |
| scaffold | measured_chronos | joint_no_expression_signature/elasticnet | 5 | 0.51134 | 0.40240 | 0.24156 |
| scaffold | measured_chronos | joint_no_expression_signature/lightgbm | 5 | 0.51984 | 0.42070 | 0.21941 |
| scaffold | measured_chronos | joint_no_expression_signature/mlp | 5 | 0.50862 | 0.39078 | 0.22355 |
| scaffold | measured_chronos | joint_no_expression_signature/ridge | 5 | 0.51331 | 0.39258 | 0.22946 |
| scaffold | measured_chronos | joint_observed/elasticnet | 5 | 0.51002 | 0.40147 | 0.24850 |
| scaffold | measured_chronos | joint_observed/lightgbm | 5 | 0.51956 | 0.42861 | 0.21773 |
| scaffold | measured_chronos | joint_observed/mlp | 5 | 0.49732 | 0.41098 | 0.22181 |
| scaffold | measured_chronos | joint_observed/ridge | 5 | 0.51344 | 0.41260 | 0.23172 |
| scaffold | measured_chronos | joint_predicted/elasticnet | 5 | 0.51108 | 0.38569 | 0.24878 |
| scaffold | measured_chronos | joint_predicted/lightgbm | 5 | 0.52255 | 0.41346 | 0.21705 |
| scaffold | measured_chronos | joint_predicted/mlp | 5 | 0.50133 | 0.44732 | 0.21883 |
| scaffold | measured_chronos | joint_predicted/ridge | 5 | 0.51359 | 0.44177 | 0.23186 |
| scaffold | measured_chronos | max_loss | 5 | 0.06454 | 0.54104 | 1.22013 |
| scaffold | measured_chronos | molformer/elasticnet | 5 | 0.51144 | 0.43613 | 0.23933 |
| scaffold | measured_chronos | molformer/lightgbm | 5 | 0.51379 | 0.46433 | 0.21653 |
| scaffold | measured_chronos | molformer/mlp | 5 | 0.49483 | 0.34758 | 0.22781 |
| scaffold | measured_chronos | molformer/ridge | 5 | 0.51167 | 0.41176 | 0.22943 |
| scaffold | measured_chronos | train_mean | 5 | 0.00000 | 0.41133 | 0.21305 |
| unseen_target | all | ABCB1 | 5 | 0.05107 | 0.41793 | 5.38262 |
| unseen_target | all | ecfp4/elasticnet | 5 | 0.49909 | 0.38317 | 0.22861 |
| unseen_target | all | ecfp4/lightgbm | 5 | 0.45500 | 0.41356 | 0.22819 |
| unseen_target | all | ecfp4/mlp | 5 | 0.48331 | 0.37703 | 0.21769 |
| unseen_target | all | ecfp4/ridge | 5 | 0.49862 | 0.37760 | 0.20817 |
| unseen_target | all | genetic_observed/elasticnet | 5 | 0.48323 | 0.46162 | 0.23196 |
| unseen_target | all | genetic_observed/lightgbm | 5 | 0.41993 | 0.44393 | 0.23242 |
| unseen_target | all | genetic_observed/mlp | 5 | 0.47993 | 0.47002 | 0.20789 |
| unseen_target | all | genetic_observed/ridge | 5 | 0.49168 | 0.43745 | 0.21177 |
| unseen_target | all | joint_no_chronos/elasticnet | 5 | 0.49909 | 0.48769 | 0.22248 |
| unseen_target | all | joint_no_chronos/lightgbm | 5 | 0.47359 | 0.43090 | 0.22235 |
| unseen_target | all | joint_no_chronos/mlp | 5 | 0.47038 | 0.35355 | 0.23061 |
| unseen_target | all | joint_no_chronos/ridge | 5 | 0.49827 | 0.41724 | 0.22614 |
| unseen_target | all | joint_no_expression_signature/elasticnet | 5 | 0.49727 | 0.47666 | 0.23186 |
| unseen_target | all | joint_no_expression_signature/lightgbm | 5 | 0.47769 | 0.42118 | 0.21631 |
| unseen_target | all | joint_no_expression_signature/mlp | 5 | 0.44720 | 0.40102 | 0.22703 |
| unseen_target | all | joint_no_expression_signature/ridge | 5 | 0.49799 | 0.42048 | 0.22673 |
| unseen_target | all | joint_observed/elasticnet | 5 | 0.49723 | 0.47723 | 0.23195 |
| unseen_target | all | joint_observed/lightgbm | 5 | 0.47492 | 0.42317 | 0.21720 |
| unseen_target | all | joint_observed/mlp | 5 | 0.46664 | 0.42699 | 0.22594 |
| unseen_target | all | joint_observed/ridge | 5 | 0.49796 | 0.42265 | 0.22662 |
| unseen_target | all | joint_predicted/elasticnet | 5 | 0.49717 | 0.45920 | 0.23606 |
| unseen_target | all | joint_predicted/lightgbm | 5 | 0.47492 | 0.42317 | 0.21720 |
| unseen_target | all | joint_predicted/mlp | 5 | 0.46991 | 0.43297 | 0.22445 |
| unseen_target | all | joint_predicted/ridge | 5 | 0.49801 | 0.42266 | 0.22621 |
| unseen_target | all | max_loss | 5 | 0.04043 | 0.54550 | 1.11476 |
| unseen_target | all | molformer/elasticnet | 5 | 0.49909 | 0.48769 | 0.22255 |
| unseen_target | all | molformer/lightgbm | 5 | 0.47359 | 0.43090 | 0.22235 |
| unseen_target | all | molformer/mlp | 5 | 0.44764 | 0.40427 | 0.22732 |
| unseen_target | all | molformer/ridge | 5 | 0.49862 | 0.41262 | 0.22614 |
| unseen_target | all | train_mean | 5 | 0.00000 | 0.41793 | 0.21544 |
| unseen_target | measured_chronos | ABCB1 | 5 | 0.06402 | 0.41133 | 4.38279 |
| unseen_target | measured_chronos | ecfp4/elasticnet | 5 | 0.47835 | 0.37866 | 0.23108 |
| unseen_target | measured_chronos | ecfp4/lightgbm | 5 | 0.44102 | 0.41074 | 0.23181 |
| unseen_target | measured_chronos | ecfp4/mlp | 5 | 0.46843 | 0.39752 | 0.22205 |
| unseen_target | measured_chronos | ecfp4/ridge | 5 | 0.47463 | 0.36096 | 0.21085 |
| unseen_target | measured_chronos | genetic_observed/elasticnet | 5 | 0.43287 | 0.46222 | 0.23656 |
| unseen_target | measured_chronos | genetic_observed/lightgbm | 5 | 0.38601 | 0.44287 | 0.23436 |
| unseen_target | measured_chronos | genetic_observed/mlp | 5 | 0.47401 | 0.48016 | 0.21088 |
| unseen_target | measured_chronos | genetic_observed/ridge | 5 | 0.45656 | 0.41777 | 0.21639 |
| unseen_target | measured_chronos | joint_no_chronos/elasticnet | 5 | 0.47835 | 0.47011 | 0.22803 |
| unseen_target | measured_chronos | joint_no_chronos/lightgbm | 5 | 0.44728 | 0.40500 | 0.23095 |
| unseen_target | measured_chronos | joint_no_chronos/mlp | 5 | 0.44435 | 0.35589 | 0.23300 |
| unseen_target | measured_chronos | joint_no_chronos/ridge | 5 | 0.47476 | 0.40345 | 0.22684 |
| unseen_target | measured_chronos | joint_no_expression_signature/elasticnet | 5 | 0.46740 | 0.47219 | 0.23701 |
| unseen_target | measured_chronos | joint_no_expression_signature/lightgbm | 5 | 0.45124 | 0.41327 | 0.22249 |
| unseen_target | measured_chronos | joint_no_expression_signature/mlp | 5 | 0.41164 | 0.39801 | 0.22997 |
| unseen_target | measured_chronos | joint_no_expression_signature/ridge | 5 | 0.47339 | 0.40392 | 0.22738 |
| unseen_target | measured_chronos | joint_observed/elasticnet | 5 | 0.46710 | 0.47210 | 0.23717 |
| unseen_target | measured_chronos | joint_observed/lightgbm | 5 | 0.44794 | 0.41080 | 0.22263 |
| unseen_target | measured_chronos | joint_observed/mlp | 5 | 0.46006 | 0.44175 | 0.22674 |
| unseen_target | measured_chronos | joint_observed/ridge | 5 | 0.47344 | 0.40573 | 0.22735 |
| unseen_target | measured_chronos | joint_predicted/elasticnet | 5 | 0.46906 | 0.45773 | 0.23807 |
| unseen_target | measured_chronos | joint_predicted/lightgbm | 5 | 0.44794 | 0.41080 | 0.22263 |
| unseen_target | measured_chronos | joint_predicted/mlp | 5 | 0.46183 | 0.43578 | 0.22612 |
| unseen_target | measured_chronos | joint_predicted/ridge | 5 | 0.47343 | 0.40594 | 0.22699 |
| unseen_target | measured_chronos | max_loss | 5 | 0.06454 | 0.54104 | 1.22013 |
| unseen_target | measured_chronos | molformer/elasticnet | 5 | 0.47835 | 0.47011 | 0.22816 |
| unseen_target | measured_chronos | molformer/lightgbm | 5 | 0.44728 | 0.40500 | 0.23095 |
| unseen_target | measured_chronos | molformer/mlp | 5 | 0.42413 | 0.40040 | 0.23120 |
| unseen_target | measured_chronos | molformer/ridge | 5 | 0.47463 | 0.39843 | 0.22677 |
| unseen_target | measured_chronos | train_mean | 5 | 0.00000 | 0.41133 | 0.21442 |
