# Track 6 — foundation representation benchmark

Scores are from separately verified frozen predictions. All means weight panel folds equally. Counts of scorable folds and assay groups are recorded in the TSVs.

## Feature extraction

| store | requested | embedded | dimension | failure_rate | device | seconds | peak VRAM MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bulk-v2/geneformer | 1064 | 1064 | 768 | 0.0000 | NVIDIA A30 | 171.6807 | 2517.1562 |
| bulk-v2/scgpt | 1064 | 1064 | 512 | 0.0000 | NVIDIA A30 | 26.4814 | 350.0933 |
| chemistry-v1/chemberta2 | 1249 | 1249 | 384 | 0.0000 | NVIDIA A30 | 5.3871 | 147.9570 |
| chemistry-v1/clamp | 1249 | 1249 | 768 | 0.0000 | NVIDIA A30 | 6.8190 | 178.6514 |
| chemistry-v1/ecfp4 | 1249 | 1249 | 2057 | 0.0000 | cpu | 1.3786 | 0.0000 |
| chemistry-v1/molformer | 1249 | 1249 | 768 | 0.0000 | NVIDIA A30 | 9.3017 | 460.9448 |
| chemistry-v1/unimol | 1249 | 1223 | 512 | 0.0208 | NVIDIA A30 | 3020.4175 | 3437.6523 |
| limore-v2/geneformer | 77 | 77 | 768 | 0.0000 | NVIDIA A30 | 17.6135 | 2517.1562 |
| limore-v2/scgpt | 77 | 77 | 512 | 0.0000 | NVIDIA A30 | 3.8111 | 350.0933 |
| protein-v1/esm2 | 20281 | 20281 | 480 | 0.0000 | NVIDIA A30 | 232.2200 | 1399.0020 |
| protein-v1/string | 15955 | 15955 | 96 | 0.0000 | cpu | 11.0871 | 0.0000 |

## Highest measured performance by endpoint

These rows identify the strongest observed fold-average scores. All head hyperparameters were chosen within their source folds before forecasting. The matched-head gain table specifies the ablation differences against ECFP4 under each identical head.

| task | metric | representation | head | score | best_ecfp4 | absolute_gain |
| --- | --- | --- | --- | --- | --- | --- |
| dili-release-increment | auprc | clamp | ridge | 0.5902 | 0.5153 | 0.0749 |
| dili-release-increment | auroc | clamp | lightgbm | 0.7514 | 0.6885 | 0.0630 |
| dili-scaffold | auprc | ecfp4+clamp | elasticnet | 0.8228 | 0.7798 | 0.0430 |
| dili-scaffold | auroc | ecfp4+clamp | elasticnet | 0.7478 | 0.7081 | 0.0397 |
| gdsc2-drug-cold | cell_macro_ndcg5 | ecfp4+chemberta2 | elasticnet | 0.7308 | 0.6443 | 0.0865 |
| gdsc2-drug-cold | within_drug_spearman | ecfp4+rank+geneformer | ridge | 0.3938 | 0.3510 | 0.0428 |
| limore-cell-cold | cell_macro_ndcg5 | ecfp4+geneformer | elasticnet | 0.5686 | 0.5448 | 0.0238 |
| limore-cell-cold | within_drug_spearman | ecfp4+paths | elasticnet | 0.3582 | 0.1920 | 0.1663 |
| limore-drug-cold | cell_macro_ndcg5 | clamp | ridge | 0.5822 | 0.5214 | 0.0607 |
| limore-drug-cold | within_drug_spearman | ecfp4+chemberta2 | ridge | 0.5286 | 0.5283 | 0.0003 |

## limore-cell-cold

1 panels, 3,400 predictions scored.

| representation | head | within_drug_spearman | cell_macro_ndcg5 |
| --- | --- | --- | --- |
| abcb1 | baseline | 0.0726 | 0.2940 |
| cellhit_xgboost | public family | 0.1305 | 0.5216 |
| chemberta2 | elasticnet | 0.1909 | 0.5679 |
| chemberta2 | lightgbm | 0.1170 | 0.5379 |
| chemberta2 | mlp | -0.0013 | 0.5509 |
| chemberta2 | ridge | 0.1468 | 0.5338 |
| clamp | elasticnet | 0.1930 | 0.5679 |
| clamp | lightgbm | 0.1904 | 0.5186 |
| clamp | mlp | 0.1172 | 0.5370 |
| clamp | ridge | 0.1475 | 0.5338 |
| drug_mean | baseline | 0.0000 | 0.4702 |
| ecfp4 | elasticnet | 0.1920 | 0.5345 |
| ecfp4 | lightgbm | 0.1311 | 0.5448 |
| ecfp4 | mlp | 0.1380 | 0.5412 |
| ecfp4 | ridge | 0.1476 | 0.5345 |
| ecfp4+chemberta2 | elasticnet | 0.1920 | 0.5345 |
| ecfp4+chemberta2 | lightgbm | 0.1450 | 0.5348 |
| ecfp4+chemberta2 | mlp | 0.0938 | 0.5299 |
| ecfp4+chemberta2 | ridge | 0.1476 | 0.5338 |
| ecfp4+clamp | elasticnet | 0.1920 | 0.5345 |
| ecfp4+clamp | lightgbm | 0.1125 | 0.5110 |
| ecfp4+clamp | mlp | 0.0592 | 0.5660 |
| ecfp4+clamp | ridge | 0.1489 | 0.5338 |
| ecfp4+geneformer | elasticnet | 0.1709 | 0.5686 |
| ecfp4+geneformer | lightgbm | 0.1589 | 0.5287 |
| ecfp4+geneformer | mlp | 0.0264 | 0.5311 |
| ecfp4+geneformer | ridge | 0.1925 | 0.5345 |
| ecfp4+molformer | elasticnet | 0.1920 | 0.5345 |
| ecfp4+molformer | lightgbm | 0.1055 | 0.5232 |
| ecfp4+molformer | mlp | 0.0918 | 0.5524 |
| ecfp4+molformer | ridge | 0.1494 | 0.5338 |
| ecfp4+paths | elasticnet | 0.3582 | 0.5345 |
| ecfp4+paths | lightgbm | 0.0922 | 0.5172 |
| ecfp4+paths | mlp | 0.0923 | 0.4900 |
| ecfp4+paths | ridge | 0.3255 | 0.5338 |
| ecfp4+rank+geneformer | elasticnet | 0.2407 | 0.5345 |
| ecfp4+rank+geneformer | lightgbm | 0.1598 | 0.5418 |
| ecfp4+rank+geneformer | mlp | 0.0719 | 0.5263 |
| ecfp4+rank+geneformer | ridge | 0.1706 | 0.5345 |
| ecfp4+rank+scgpt | elasticnet | 0.0436 | 0.5686 |
| ecfp4+rank+scgpt | lightgbm | 0.0655 | 0.5433 |
| ecfp4+rank+scgpt | mlp | 0.1709 | 0.5179 |
| ecfp4+rank+scgpt | ridge | 0.1988 | 0.5345 |
| ecfp4+scgpt | elasticnet | 0.1161 | 0.5686 |
| ecfp4+scgpt | lightgbm | 0.1206 | 0.5349 |
| ecfp4+scgpt | mlp | 0.0609 | 0.5145 |
| ecfp4+scgpt | ridge | 0.2150 | 0.5338 |
| ecfp4+unimol | elasticnet | 0.1917 | 0.5345 |
| ecfp4+unimol | lightgbm | 0.0848 | 0.5160 |
| ecfp4+unimol | mlp | 0.1888 | 0.5523 |
| ecfp4+unimol | ridge | 0.1498 | 0.4702 |
| mean | baseline | 0.0000 | 0.2940 |
| mki67 | baseline | 0.1871 | 0.2940 |
| molformer | elasticnet | 0.1929 | 0.5338 |
| molformer | lightgbm | 0.1233 | 0.5196 |
| molformer | mlp | 0.0896 | 0.5387 |
| molformer | ridge | 0.1494 | 0.5338 |
| per_drug_ridge | baseline | 0.1730 | 0.5222 |
| unimol | elasticnet | 0.1937 | 0.5042 |
| unimol | lightgbm | 0.0644 | 0.5046 |
| unimol | mlp | 0.1362 | 0.5295 |
| unimol | ridge | 0.1498 | 0.5042 |

## limore-drug-cold

5 panels, 6,776 predictions scored.

| representation | head | within_drug_spearman | cell_macro_ndcg5 |
| --- | --- | --- | --- |
| abcb1 | baseline | 0.0616 | 0.4146 |
| chemberta2 | elasticnet | 0.5220 | 0.3903 |
| chemberta2 | lightgbm | 0.5111 | 0.3807 |
| chemberta2 | mlp | 0.3686 | 0.4716 |
| chemberta2 | ridge | 0.5277 | 0.4326 |
| clamp | elasticnet | 0.5246 | 0.5502 |
| clamp | lightgbm | 0.5092 | 0.5039 |
| clamp | mlp | 0.3808 | 0.4641 |
| clamp | ridge | 0.5284 | 0.5822 |
| drug_mean | baseline | 0.0000 | 0.4146 |
| ecfp4 | elasticnet | 0.5163 | 0.5214 |
| ecfp4 | lightgbm | 0.5157 | 0.4396 |
| ecfp4 | mlp | 0.4497 | 0.4497 |
| ecfp4 | ridge | 0.5283 | 0.5090 |
| ecfp4+chemberta2 | elasticnet | 0.5245 | 0.4992 |
| ecfp4+chemberta2 | lightgbm | 0.5156 | 0.4211 |
| ecfp4+chemberta2 | mlp | 0.4453 | 0.4536 |
| ecfp4+chemberta2 | ridge | 0.5286 | 0.4716 |
| ecfp4+clamp | elasticnet | 0.5212 | 0.5522 |
| ecfp4+clamp | lightgbm | 0.5195 | 0.4974 |
| ecfp4+clamp | mlp | 0.4448 | 0.4538 |
| ecfp4+clamp | ridge | 0.5279 | 0.4911 |
| ecfp4+geneformer | elasticnet | 0.5091 | 0.5214 |
| ecfp4+geneformer | lightgbm | 0.5132 | 0.4317 |
| ecfp4+geneformer | mlp | 0.4564 | 0.4537 |
| ecfp4+geneformer | ridge | 0.5276 | 0.5090 |
| ecfp4+molformer | elasticnet | 0.5263 | 0.4667 |
| ecfp4+molformer | lightgbm | 0.5049 | 0.4815 |
| ecfp4+molformer | mlp | 0.4422 | 0.4398 |
| ecfp4+molformer | ridge | 0.5281 | 0.4743 |
| ecfp4+paths | elasticnet | 0.5014 | 0.5214 |
| ecfp4+paths | lightgbm | 0.5145 | 0.4245 |
| ecfp4+paths | mlp | 0.4439 | 0.4740 |
| ecfp4+paths | ridge | 0.5269 | 0.5090 |
| ecfp4+rank+geneformer | elasticnet | 0.5196 | 0.5214 |
| ecfp4+rank+geneformer | lightgbm | 0.5173 | 0.4295 |
| ecfp4+rank+geneformer | mlp | 0.4555 | 0.4947 |
| ecfp4+rank+geneformer | ridge | 0.5283 | 0.5090 |
| ecfp4+rank+scgpt | elasticnet | 0.5166 | 0.5214 |
| ecfp4+rank+scgpt | lightgbm | 0.5179 | 0.4151 |
| ecfp4+rank+scgpt | mlp | 0.4470 | 0.4709 |
| ecfp4+rank+scgpt | ridge | 0.5283 | 0.5090 |
| ecfp4+scgpt | elasticnet | 0.4744 | 0.5214 |
| ecfp4+scgpt | lightgbm | 0.5231 | 0.4408 |
| ecfp4+scgpt | mlp | 0.4496 | 0.4477 |
| ecfp4+scgpt | ridge | 0.5230 | 0.5090 |
| ecfp4+unimol | elasticnet | 0.5220 | 0.4713 |
| ecfp4+unimol | lightgbm | 0.5070 | 0.4146 |
| ecfp4+unimol | mlp | 0.4264 | 0.4325 |
| ecfp4+unimol | ridge | 0.5286 | 0.4713 |
| mean | baseline | 0.0000 | 0.4146 |
| mki67 | baseline | 0.1631 | 0.4146 |
| molformer | elasticnet | 0.5163 | 0.4284 |
| molformer | lightgbm | 0.5061 | 0.4983 |
| molformer | mlp | 0.4092 | 0.3633 |
| molformer | ridge | 0.5276 | 0.4374 |
| unimol | elasticnet | 0.5238 | 0.3128 |
| unimol | lightgbm | 0.4984 | 0.3697 |
| unimol | mlp | 0.3991 | 0.3882 |
| unimol | ridge | 0.5284 | 0.3280 |

## gdsc2-drug-cold

5 panels, 24,839 predictions scored.

| representation | head | within_drug_spearman | cell_macro_ndcg5 |
| --- | --- | --- | --- |
| abcb1 | baseline | 0.0296 | 0.6421 |
| chemberta2 | elasticnet | 0.2665 | 0.6314 |
| chemberta2 | lightgbm | 0.3459 | 0.6847 |
| chemberta2 | mlp | 0.2828 | 0.6894 |
| chemberta2 | ridge | 0.3412 | 0.6349 |
| clamp | elasticnet | 0.2984 | 0.5979 |
| clamp | lightgbm | 0.3581 | 0.5978 |
| clamp | mlp | 0.2985 | 0.6993 |
| clamp | ridge | 0.3473 | 0.6638 |
| drug_mean | baseline | 0.0000 | 0.6421 |
| ecfp4 | elasticnet | 0.2984 | 0.5873 |
| ecfp4 | lightgbm | 0.3510 | 0.6205 |
| ecfp4 | mlp | 0.3391 | 0.6443 |
| ecfp4 | ridge | 0.3473 | 0.6209 |
| ecfp4+chemberta2 | elasticnet | 0.3098 | 0.7308 |
| ecfp4+chemberta2 | lightgbm | 0.3512 | 0.6894 |
| ecfp4+chemberta2 | mlp | 0.3380 | 0.6700 |
| ecfp4+chemberta2 | ridge | 0.3473 | 0.6416 |
| ecfp4+clamp | elasticnet | 0.2984 | 0.6006 |
| ecfp4+clamp | lightgbm | 0.3528 | 0.5954 |
| ecfp4+clamp | mlp | 0.3326 | 0.7047 |
| ecfp4+clamp | ridge | 0.3473 | 0.6519 |
| ecfp4+geneformer | elasticnet | 0.2839 | 0.5873 |
| ecfp4+geneformer | lightgbm | 0.3568 | 0.6277 |
| ecfp4+geneformer | mlp | 0.3462 | 0.6442 |
| ecfp4+geneformer | ridge | 0.3832 | 0.6209 |
| ecfp4+molformer | elasticnet | 0.2640 | 0.5392 |
| ecfp4+molformer | lightgbm | 0.3441 | 0.5995 |
| ecfp4+molformer | mlp | 0.3324 | 0.6277 |
| ecfp4+molformer | ridge | 0.3473 | 0.5702 |
| ecfp4+paths | elasticnet | 0.2790 | 0.5873 |
| ecfp4+paths | lightgbm | 0.3642 | 0.6468 |
| ecfp4+paths | mlp | 0.3345 | 0.6590 |
| ecfp4+paths | ridge | 0.3185 | 0.6209 |
| ecfp4+rank+geneformer | elasticnet | 0.3062 | 0.5873 |
| ecfp4+rank+geneformer | lightgbm | 0.3683 | 0.6321 |
| ecfp4+rank+geneformer | mlp | 0.3464 | 0.6340 |
| ecfp4+rank+geneformer | ridge | 0.3938 | 0.6209 |
| ecfp4+rank+scgpt | elasticnet | 0.2908 | 0.5873 |
| ecfp4+rank+scgpt | lightgbm | 0.3486 | 0.6255 |
| ecfp4+rank+scgpt | mlp | 0.3366 | 0.6322 |
| ecfp4+rank+scgpt | ridge | 0.3926 | 0.6209 |
| ecfp4+scgpt | elasticnet | 0.2513 | 0.6081 |
| ecfp4+scgpt | lightgbm | 0.3607 | 0.6467 |
| ecfp4+scgpt | mlp | 0.3359 | 0.6356 |
| ecfp4+scgpt | ridge | 0.3564 | 0.6209 |
| ecfp4+unimol | elasticnet | 0.2762 | 0.5636 |
| ecfp4+unimol | lightgbm | 0.3651 | 0.6020 |
| ecfp4+unimol | mlp | 0.3180 | 0.6571 |
| ecfp4+unimol | ridge | 0.3473 | 0.5875 |
| mean | baseline | 0.0000 | 0.6421 |
| mki67 | baseline | 0.0264 | 0.6421 |
| molformer | elasticnet | 0.2865 | 0.5628 |
| molformer | lightgbm | 0.3389 | 0.6009 |
| molformer | mlp | 0.3123 | 0.6288 |
| molformer | ridge | 0.3473 | 0.5730 |
| unimol | elasticnet | 0.2983 | 0.5438 |
| unimol | lightgbm | 0.3606 | 0.5951 |
| unimol | mlp | 0.2708 | 0.6281 |
| unimol | ridge | 0.3473 | 0.5840 |

## dili-scaffold

5 panels, 858 predictions scored.

| representation | head | auroc | auprc |
| --- | --- | --- | --- |
| chemberta2 | elasticnet | 0.7013 | 0.7656 |
| chemberta2 | lightgbm | 0.7348 | 0.7869 |
| chemberta2 | mlp | 0.7005 | 0.7659 |
| chemberta2 | ridge | 0.6967 | 0.7679 |
| chemprop_dmpnn | public family | 0.6805 | 0.7464 |
| clamp | elasticnet | 0.7375 | 0.8202 |
| clamp | lightgbm | 0.7251 | 0.8002 |
| clamp | mlp | 0.7001 | 0.7795 |
| clamp | ridge | 0.7027 | 0.7716 |
| clogp | baseline | 0.5549 | 0.6564 |
| ecfp4 | elasticnet | 0.6647 | 0.7366 |
| ecfp4 | lightgbm | 0.7081 | 0.7798 |
| ecfp4 | mlp | 0.6840 | 0.7712 |
| ecfp4 | ridge | 0.6335 | 0.7214 |
| ecfp4+chemberta2 | elasticnet | 0.6995 | 0.7491 |
| ecfp4+chemberta2 | lightgbm | 0.7401 | 0.7921 |
| ecfp4+chemberta2 | mlp | 0.6967 | 0.7590 |
| ecfp4+chemberta2 | ridge | 0.6452 | 0.7212 |
| ecfp4+clamp | elasticnet | 0.7478 | 0.8228 |
| ecfp4+clamp | lightgbm | 0.7130 | 0.7928 |
| ecfp4+clamp | mlp | 0.7360 | 0.8005 |
| ecfp4+clamp | ridge | 0.6465 | 0.7240 |
| ecfp4+molformer | elasticnet | 0.6928 | 0.7637 |
| ecfp4+molformer | lightgbm | 0.7001 | 0.7732 |
| ecfp4+molformer | mlp | 0.7183 | 0.7798 |
| ecfp4+molformer | ridge | 0.6573 | 0.7284 |
| ecfp4+unimol | elasticnet | 0.6958 | 0.7535 |
| ecfp4+unimol | lightgbm | 0.6992 | 0.7663 |
| ecfp4+unimol | mlp | 0.7065 | 0.7681 |
| ecfp4+unimol | ridge | 0.6428 | 0.7187 |
| mean | baseline | 0.5000 | 0.6191 |
| molformer | elasticnet | 0.6685 | 0.7546 |
| molformer | lightgbm | 0.7030 | 0.7636 |
| molformer | mlp | 0.7028 | 0.7765 |
| molformer | ridge | 0.6806 | 0.7601 |
| mw_logp_rule | baseline | 0.5110 | 0.6273 |
| unimol | elasticnet | 0.6852 | 0.7463 |
| unimol | lightgbm | 0.6896 | 0.7642 |
| unimol | mlp | 0.6720 | 0.7425 |
| unimol | ridge | 0.6923 | 0.7427 |

## dili-release-increment

1 panels, 147 predictions scored.

| representation | head | auroc | auprc |
| --- | --- | --- | --- |
| chemberta2 | elasticnet | 0.6808 | 0.4302 |
| chemberta2 | lightgbm | 0.7026 | 0.5288 |
| chemberta2 | mlp | 0.6664 | 0.4564 |
| chemberta2 | ridge | 0.6630 | 0.4288 |
| chemprop_dmpnn | public family | 0.6845 | 0.4516 |
| clamp | elasticnet | 0.7087 | 0.5737 |
| clamp | lightgbm | 0.7514 | 0.5117 |
| clamp | mlp | 0.7081 | 0.5067 |
| clamp | ridge | 0.7368 | 0.5902 |
| clogp | baseline | 0.6987 | 0.4989 |
| ecfp4 | elasticnet | 0.6205 | 0.3542 |
| ecfp4 | lightgbm | 0.6728 | 0.5153 |
| ecfp4 | mlp | 0.6885 | 0.4379 |
| ecfp4 | ridge | 0.6046 | 0.3726 |
| ecfp4+chemberta2 | elasticnet | 0.6288 | 0.3817 |
| ecfp4+chemberta2 | lightgbm | 0.6954 | 0.4901 |
| ecfp4+chemberta2 | mlp | 0.6863 | 0.4359 |
| ecfp4+chemberta2 | ridge | 0.5712 | 0.3501 |
| ecfp4+clamp | elasticnet | 0.7464 | 0.5568 |
| ecfp4+clamp | lightgbm | 0.7477 | 0.5091 |
| ecfp4+clamp | mlp | 0.6922 | 0.4860 |
| ecfp4+clamp | ridge | 0.5782 | 0.3800 |
| ecfp4+molformer | elasticnet | 0.6364 | 0.3809 |
| ecfp4+molformer | lightgbm | 0.6854 | 0.4429 |
| ecfp4+molformer | mlp | 0.7004 | 0.4660 |
| ecfp4+molformer | ridge | 0.6294 | 0.3852 |
| ecfp4+unimol | elasticnet | 0.6246 | 0.4058 |
| ecfp4+unimol | lightgbm | 0.6312 | 0.3999 |
| ecfp4+unimol | mlp | 0.6525 | 0.4209 |
| ecfp4+unimol | ridge | 0.5767 | 0.3482 |
| mean | baseline | 0.5000 | 0.3061 |
| molformer | elasticnet | 0.6989 | 0.5569 |
| molformer | lightgbm | 0.7035 | 0.5485 |
| molformer | mlp | 0.6991 | 0.5492 |
| molformer | ridge | 0.7203 | 0.5366 |
| mw_logp_rule | baseline | 0.6147 | 0.3756 |
| unimol | elasticnet | 0.6874 | 0.4380 |
| unimol | lightgbm | 0.6327 | 0.4261 |
| unimol | mlp | 0.6723 | 0.4603 |
| unimol | ridge | 0.6874 | 0.4216 |

## Public model families

CellHit-style per-drug XGBoost was refit on normalized source AUC and evaluated on the identical LIMORE external cells. The published model uses assay-specific ln(IC50) and log-fold change and is unavailable for genuinely unseen compounds. The Chemprop-style D-MPNN was independently refit on identical DILI scaffold and release-increment partitions. The two model families share the benchmark's test IDs and label hashes.

## External 3D data

The LICOB RNA workbook failed the repository checksum and workbook parser. The LIMORE cell-cold table is the designated 2D fallback and contributes no standalone 3D score.
