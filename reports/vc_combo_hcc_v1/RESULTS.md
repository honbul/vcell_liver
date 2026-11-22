# Track 8 — held adult HCC drug-combination assays

Every model was selected using its inner blocked validation. Frozen predictions were verified against held assay IDs and outcome hashes in a separate command.

The Sanger scorecard includes measured Bliss, reconstructed ZIP and cross-design CSS, and maximum growth inhibition gained over the strongest measured single agent. Sanger CSS requires both relative single-agent IC50 values to be bracketed by measured doses. DrugComb's published CSS is a separate protocol. A missing score indicates unsupported held observations.

## lco_hcc

| source | endpoint | model | n | lineages | spearman | rmse | cell_macro_spearman | cell_macro_spearman_low | cell_macro_spearman_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sanger | zip | cross_attention | 1114 | 11 | 0.5444 | 4.2024 | 0.5389 | 0.4870 | 0.5904 |
| Sanger | bliss | cross_attention | 1114 | 11 | 0.5220 | 3.7466 | 0.5243 | 0.4547 | 0.5854 |
| Sanger | css | cross_attention | 249 | 11 | 0.6278 | 13.7017 | 0.5800 | 0.4696 | 0.6823 |
| Sanger | added_inhibition | cross_attention | 1114 | 11 | 0.6136 | 10.6855 | 0.6203 | 0.5579 | 0.6856 |
| Sanger | zip | deepsynergy | 1114 | 11 | 0.5329 | 4.3957 | 0.5345 | 0.4723 | 0.5939 |
| Sanger | bliss | deepsynergy | 1114 | 11 | 0.5036 | 3.9320 | 0.5242 | 0.4688 | 0.5734 |
| Sanger | css | deepsynergy | 249 | 11 | 0.6218 | 13.1283 | 0.5516 | 0.4562 | 0.6482 |
| Sanger | added_inhibition | deepsynergy | 1114 | 11 | 0.5994 | 10.6766 | 0.6152 | 0.5639 | 0.6694 |
| Sanger | zip | elasticnet | 1114 | 11 | 0.4080 | 4.7061 | 0.4984 | 0.4461 | 0.5567 |
| Sanger | bliss | elasticnet | 1114 | 11 | 0.2785 | 5.2107 | 0.4519 | 0.4003 | 0.5037 |
| Sanger | css | elasticnet | 249 | 11 | 0.6279 | 15.3060 | 0.5674 | 0.4536 | 0.6837 |
| Sanger | added_inhibition | elasticnet | 1114 | 11 | 0.5145 | 11.6458 | 0.5433 | 0.4538 | 0.6203 |
| Sanger | zip | lightgbm | 1114 | 11 | 0.5352 | 4.3121 | 0.5218 | 0.4647 | 0.5877 |
| Sanger | bliss | lightgbm | 1114 | 11 | 0.4544 | 3.9585 | 0.4775 | 0.4086 | 0.5313 |
| Sanger | css | lightgbm | 249 | 11 | 0.5967 | 16.0389 | 0.5613 | 0.4652 | 0.6597 |
| Sanger | added_inhibition | lightgbm | 1114 | 11 | 0.5908 | 11.0041 | 0.5914 | 0.5143 | 0.6689 |
| Sanger | zip | matchmaker | 1114 | 11 | 0.5257 | 4.3162 | 0.5320 | 0.4765 | 0.5849 |
| Sanger | bliss | matchmaker | 1114 | 11 | 0.4991 | 3.9162 | 0.5045 | 0.4360 | 0.5645 |
| Sanger | css | matchmaker | 249 | 11 | 0.6342 | 13.0211 | 0.5109 | 0.3929 | 0.6281 |
| Sanger | added_inhibition | matchmaker | 1114 | 11 | 0.6194 | 10.3788 | 0.6327 | 0.5730 | 0.6886 |
| Sanger | zip | mean | 1114 | 11 | 0.1926 | 5.0659 | 0.1783 | 0.0846 | 0.2543 |
| Sanger | bliss | mean | 1114 | 11 | 0.1405 | 4.5256 | 0.1706 | 0.1226 | 0.2197 |
| Sanger | css | mean | 249 | 11 | 0.1802 | 22.3933 | 0.1339 | -0.0706 | 0.3115 |
| Sanger | added_inhibition | mean | 1114 | 11 | 0.1109 | 14.1058 | 0.0820 | -0.0199 | 0.1642 |
| Sanger | zip | mono_bliss_independent | 1114 | 11 | 0.0000 | 5.5204 | 0.0000 | 0.0000 | 0.0000 |
| Sanger | bliss | mono_bliss_independent | 1114 | 11 | 0.0000 | 4.7696 | 0.0000 | 0.0000 | 0.0000 |
| Sanger | css | mono_bliss_independent | 74 | 10 | 0.5125 | 15.8809 | 0.3225 | 0.0919 | 0.5374 |
| Sanger | added_inhibition | mono_bliss_independent | 353 | 11 | 0.2318 | 18.3596 | 0.1758 | 0.0705 | 0.2720 |
| Sanger | zip | no_crispr | 1114 | 11 | 0.5214 | 4.2888 | 0.5291 | 0.4809 | 0.5814 |
| Sanger | bliss | no_crispr | 1114 | 11 | 0.5200 | 3.8356 | 0.5403 | 0.4929 | 0.5871 |
| Sanger | css | no_crispr | 249 | 11 | 0.6400 | 14.2451 | 0.5467 | 0.4205 | 0.6635 |
| Sanger | added_inhibition | no_crispr | 1114 | 11 | 0.6182 | 10.5003 | 0.6258 | 0.5723 | 0.6856 |
| Sanger | zip | no_foundation | 1114 | 11 | 0.5207 | 4.3561 | 0.5222 | 0.4707 | 0.5705 |
| Sanger | bliss | no_foundation | 1114 | 11 | 0.4893 | 3.9831 | 0.5017 | 0.4357 | 0.5616 |
| Sanger | css | no_foundation | 249 | 11 | 0.5841 | 14.8030 | 0.5208 | 0.4030 | 0.6469 |
| Sanger | added_inhibition | no_foundation | 1114 | 11 | 0.6250 | 10.5270 | 0.6350 | 0.5783 | 0.6922 |
| Sanger | zip | no_targets | 1114 | 11 | 0.5371 | 4.2670 | 0.5405 | 0.4917 | 0.5882 |
| Sanger | bliss | no_targets | 1114 | 11 | 0.5234 | 3.8652 | 0.5357 | 0.4834 | 0.5869 |
| Sanger | css | no_targets | 249 | 11 | 0.5829 | 14.6100 | 0.5096 | 0.3967 | 0.6206 |
| Sanger | added_inhibition | no_targets | 1114 | 11 | 0.6276 | 10.3594 | 0.6343 | 0.5662 | 0.6956 |
| Sanger | zip | pair_mean | 1114 | 11 | 0.5392 | 4.3674 | 0.5259 | 0.4715 | 0.5816 |
| Sanger | bliss | pair_mean | 1114 | 11 | 0.4857 | 3.9501 | 0.4957 | 0.4413 | 0.5450 |
| Sanger | css | pair_mean | 249 | 11 | 0.5434 | 20.0945 | 0.5407 | 0.4478 | 0.6310 |
| Sanger | added_inhibition | pair_mean | 1114 | 11 | 0.5476 | 11.5880 | 0.5500 | 0.4616 | 0.6230 |
| Sanger | zip | ridge | 1114 | 11 | 0.4405 | 4.6362 | 0.5044 | 0.4472 | 0.5627 |
| Sanger | bliss | ridge | 1114 | 11 | 0.2098 | 5.9301 | 0.4617 | 0.4050 | 0.5191 |
| Sanger | css | ridge | 249 | 11 | 0.6460 | 16.0707 | 0.5901 | 0.4867 | 0.6939 |
| Sanger | added_inhibition | ridge | 1114 | 11 | 0.4710 | 12.1949 | 0.5454 | 0.4540 | 0.6196 |
| Sanger | zip | target_dependency_sum | 1114 | 11 | 0.1278 | 6.2320 | 0.1472 | 0.0854 | 0.2090 |
| Sanger | bliss | target_dependency_sum | 1114 | 11 | 0.1393 | 5.4880 | 0.1558 | 0.0917 | 0.2219 |
| Sanger | css | target_dependency_sum | 249 | 11 | 0.1598 | 25.7025 | 0.1560 | 0.0176 | 0.2944 |
| Sanger | added_inhibition | target_dependency_sum | 1114 | 11 | 0.1898 | 16.8752 | 0.2028 | 0.1544 | 0.2530 |

## lpo

| source | endpoint | model | n | lineages | spearman | rmse | cell_macro_spearman | cell_macro_spearman_low | cell_macro_spearman_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sanger | zip | cross_attention | 1114 | 11 | 0.3487 | 4.7751 | 0.3387 | 0.2803 | 0.3929 |
| Sanger | bliss | cross_attention | 1114 | 11 | 0.2069 | 4.4912 | 0.1829 | 0.0904 | 0.2744 |
| Sanger | css | cross_attention | 249 | 11 | 0.5362 | 14.1614 | 0.3185 | 0.1804 | 0.4613 |
| Sanger | added_inhibition | cross_attention | 1114 | 11 | 0.4837 | 12.1494 | 0.4736 | 0.4349 | 0.5138 |
| Sanger | zip | deepsynergy | 1114 | 11 | 0.3239 | 5.0168 | 0.2997 | 0.2539 | 0.3470 |
| Sanger | bliss | deepsynergy | 1114 | 11 | 0.2529 | 4.8268 | 0.2348 | 0.1597 | 0.3212 |
| Sanger | css | deepsynergy | 249 | 11 | 0.5940 | 14.3053 | 0.3085 | 0.0729 | 0.5104 |
| Sanger | added_inhibition | deepsynergy | 1114 | 11 | 0.5089 | 11.8829 | 0.5062 | 0.4685 | 0.5414 |
| Sanger | zip | elasticnet | 1114 | 11 | 0.2337 | 5.2391 | 0.2414 | 0.1892 | 0.2962 |
| Sanger | bliss | elasticnet | 1114 | 11 | 0.1661 | 5.1610 | 0.1745 | 0.1260 | 0.2196 |
| Sanger | css | elasticnet | 249 | 11 | 0.4048 | 17.0555 | 0.3240 | 0.1907 | 0.4638 |
| Sanger | added_inhibition | elasticnet | 1114 | 11 | 0.2732 | 13.5250 | 0.2674 | 0.1814 | 0.3430 |
| Sanger | zip | lightgbm | 1114 | 11 | 0.2088 | 5.1301 | 0.1887 | 0.1260 | 0.2489 |
| Sanger | bliss | lightgbm | 1114 | 11 | 0.0536 | 5.1598 | 0.0534 | -0.0129 | 0.1154 |
| Sanger | css | lightgbm | 249 | 11 | 0.4354 | 15.7742 | 0.1752 | -0.0749 | 0.3788 |
| Sanger | added_inhibition | lightgbm | 1114 | 11 | 0.4192 | 12.8237 | 0.4044 | 0.3619 | 0.4484 |
| Sanger | zip | matchmaker | 1114 | 11 | 0.3646 | 4.7423 | 0.3559 | 0.2969 | 0.4099 |
| Sanger | bliss | matchmaker | 1114 | 11 | 0.2585 | 4.5968 | 0.2330 | 0.1616 | 0.2982 |
| Sanger | css | matchmaker | 249 | 11 | 0.6163 | 13.5779 | 0.3560 | 0.1707 | 0.5196 |
| Sanger | added_inhibition | matchmaker | 1114 | 11 | 0.5216 | 11.6849 | 0.5126 | 0.4507 | 0.5668 |
| Sanger | zip | mean | 1114 | 11 | 0.1229 | 5.1440 | 0.1033 | 0.0283 | 0.1697 |
| Sanger | bliss | mean | 1114 | 11 | 0.0614 | 4.6047 | 0.0761 | 0.0099 | 0.1299 |
| Sanger | css | mean | 249 | 11 | 0.0074 | 22.0832 | -0.1398 | -0.3191 | 0.0184 |
| Sanger | added_inhibition | mean | 1114 | 11 | 0.0599 | 14.2911 | 0.0297 | -0.0415 | 0.0897 |
| Sanger | zip | mono_bliss_independent | 1114 | 11 | 0.0000 | 5.5204 | 0.0000 | 0.0000 | 0.0000 |
| Sanger | bliss | mono_bliss_independent | 1114 | 11 | 0.0000 | 4.7696 | 0.0000 | 0.0000 | 0.0000 |
| Sanger | css | mono_bliss_independent | 74 | 10 | 0.5125 | 15.8809 | 0.3225 | 0.0771 | 0.5431 |
| Sanger | added_inhibition | mono_bliss_independent | 353 | 11 | 0.2318 | 18.3596 | 0.1758 | 0.0680 | 0.2736 |
| Sanger | zip | no_crispr | 1114 | 11 | 0.3092 | 4.8075 | 0.3047 | 0.2584 | 0.3549 |
| Sanger | bliss | no_crispr | 1114 | 11 | 0.2378 | 4.3772 | 0.2194 | 0.1607 | 0.2744 |
| Sanger | css | no_crispr | 249 | 11 | 0.5717 | 14.1430 | 0.3539 | 0.2271 | 0.4884 |
| Sanger | added_inhibition | no_crispr | 1114 | 11 | 0.4881 | 12.0169 | 0.4809 | 0.4334 | 0.5276 |
| Sanger | zip | no_foundation | 1114 | 11 | 0.3713 | 4.6828 | 0.3647 | 0.2983 | 0.4259 |
| Sanger | bliss | no_foundation | 1114 | 11 | 0.2558 | 4.3766 | 0.2333 | 0.1555 | 0.3062 |
| Sanger | css | no_foundation | 249 | 11 | 0.5618 | 14.0415 | 0.3131 | 0.1713 | 0.4538 |
| Sanger | added_inhibition | no_foundation | 1114 | 11 | 0.5342 | 11.6461 | 0.5297 | 0.4915 | 0.5688 |
| Sanger | zip | no_targets | 1114 | 11 | 0.3459 | 4.8043 | 0.3366 | 0.2787 | 0.3927 |
| Sanger | bliss | no_targets | 1114 | 11 | 0.2212 | 4.5052 | 0.2020 | 0.1228 | 0.2772 |
| Sanger | css | no_targets | 249 | 11 | 0.5087 | 14.8455 | 0.2468 | 0.1049 | 0.4074 |
| Sanger | added_inhibition | no_targets | 1114 | 11 | 0.4665 | 12.2859 | 0.4636 | 0.4173 | 0.5058 |
| Sanger | zip | pair_mean | 1114 | 11 | 0.1229 | 5.1440 | 0.1033 | 0.0259 | 0.1684 |
| Sanger | bliss | pair_mean | 1114 | 11 | 0.0614 | 4.6047 | 0.0761 | 0.0033 | 0.1305 |
| Sanger | css | pair_mean | 249 | 11 | 0.0074 | 22.0832 | -0.1398 | -0.3168 | 0.0216 |
| Sanger | added_inhibition | pair_mean | 1114 | 11 | 0.0599 | 14.2911 | 0.0297 | -0.0400 | 0.0934 |
| Sanger | zip | ridge | 1114 | 11 | 0.2158 | 6.1785 | 0.1819 | 0.1227 | 0.2357 |
| Sanger | bliss | ridge | 1114 | 11 | 0.1695 | 6.2357 | 0.1581 | 0.1041 | 0.2062 |
| Sanger | css | ridge | 249 | 11 | 0.4759 | 17.0560 | 0.3824 | 0.2479 | 0.5135 |
| Sanger | added_inhibition | ridge | 1114 | 11 | 0.1928 | 15.5588 | 0.1827 | 0.0755 | 0.2762 |
| Sanger | zip | target_dependency_sum | 1114 | 11 | 0.1278 | 6.2320 | 0.1472 | 0.0884 | 0.2130 |
| Sanger | bliss | target_dependency_sum | 1114 | 11 | 0.1393 | 5.4880 | 0.1558 | 0.0895 | 0.2235 |
| Sanger | css | target_dependency_sum | 249 | 11 | 0.1598 | 25.7025 | 0.1560 | 0.0132 | 0.2923 |
| Sanger | added_inhibition | target_dependency_sum | 1114 | 11 | 0.1898 | 16.8752 | 0.2028 | 0.1540 | 0.2548 |

## ldo

| source | endpoint | model | n | lineages | spearman | rmse | cell_macro_spearman | cell_macro_spearman_low | cell_macro_spearman_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sanger | zip | cross_attention | 1114 | 11 | 0.0969 | 5.2587 | 0.0974 | 0.0548 | 0.1410 |
| Sanger | bliss | cross_attention | 1114 | 11 | 0.0661 | 4.9733 | 0.0522 | -0.0061 | 0.1106 |
| Sanger | css | cross_attention | 249 | 11 | 0.4751 | 15.0648 | 0.3287 | 0.1813 | 0.4608 |
| Sanger | added_inhibition | cross_attention | 1114 | 11 | 0.1980 | 15.0089 | 0.1855 | 0.1140 | 0.2608 |
| Sanger | zip | deepsynergy | 1114 | 11 | 0.0983 | 5.8165 | 0.0730 | 0.0048 | 0.1400 |
| Sanger | bliss | deepsynergy | 1114 | 11 | 0.0306 | 5.7433 | 0.0003 | -0.0794 | 0.0856 |
| Sanger | css | deepsynergy | 249 | 11 | 0.4542 | 16.0386 | 0.1929 | 0.0463 | 0.3251 |
| Sanger | added_inhibition | deepsynergy | 1114 | 11 | 0.2490 | 14.6893 | 0.2356 | 0.1526 | 0.3077 |
| Sanger | zip | elasticnet | 1114 | 11 | -0.1297 | 7.6033 | -0.1494 | -0.2148 | -0.0762 |
| Sanger | bliss | elasticnet | 1114 | 11 | -0.1202 | 7.6030 | -0.1231 | -0.1914 | -0.0501 |
| Sanger | css | elasticnet | 249 | 11 | 0.0120 | 29.7755 | 0.1364 | -0.0186 | 0.2980 |
| Sanger | added_inhibition | elasticnet | 1114 | 11 | -0.0821 | 15.7474 | -0.0981 | -0.1462 | -0.0524 |
| Sanger | zip | lightgbm | 1114 | 11 | -0.0916 | 6.0288 | -0.1198 | -0.1882 | -0.0543 |
| Sanger | bliss | lightgbm | 1114 | 11 | -0.0820 | 7.1971 | -0.0883 | -0.1527 | -0.0242 |
| Sanger | css | lightgbm | 249 | 11 | 0.1098 | 18.1202 | 0.0280 | -0.0618 | 0.1211 |
| Sanger | added_inhibition | lightgbm | 1114 | 11 | 0.0902 | 14.3669 | 0.0413 | -0.0310 | 0.1102 |
| Sanger | zip | matchmaker | 1114 | 11 | 0.0420 | 5.5788 | 0.0351 | -0.0133 | 0.0782 |
| Sanger | bliss | matchmaker | 1114 | 11 | -0.0001 | 5.2812 | 0.0002 | -0.0550 | 0.0656 |
| Sanger | css | matchmaker | 249 | 11 | 0.5407 | 16.2662 | 0.2543 | 0.1058 | 0.3947 |
| Sanger | added_inhibition | matchmaker | 1114 | 11 | 0.1858 | 16.0809 | 0.1592 | 0.0698 | 0.2462 |
| Sanger | zip | mean | 1114 | 11 | 0.0541 | 5.2105 | 0.0328 | -0.0291 | 0.0863 |
| Sanger | bliss | mean | 1114 | 11 | 0.0845 | 4.5854 | 0.1104 | 0.0670 | 0.1519 |
| Sanger | css | mean | 249 | 11 | -0.0637 | 22.9835 | -0.0736 | -0.2077 | 0.0774 |
| Sanger | added_inhibition | mean | 1114 | 11 | -0.0202 | 14.6183 | -0.0405 | -0.0977 | 0.0187 |
| Sanger | zip | mono_bliss_independent | 1114 | 11 | 0.0000 | 5.5204 | 0.0000 | 0.0000 | 0.0000 |
| Sanger | bliss | mono_bliss_independent | 1114 | 11 | 0.0000 | 4.7696 | 0.0000 | 0.0000 | 0.0000 |
| Sanger | css | mono_bliss_independent | 74 | 10 | 0.5125 | 15.8809 | 0.3225 | 0.0690 | 0.5376 |
| Sanger | added_inhibition | mono_bliss_independent | 353 | 11 | 0.2318 | 18.3596 | 0.1758 | 0.0671 | 0.2692 |
| Sanger | zip | no_crispr | 1114 | 11 | 0.1013 | 5.3778 | 0.1097 | 0.0555 | 0.1692 |
| Sanger | bliss | no_crispr | 1114 | 11 | 0.0750 | 4.9864 | 0.0618 | 0.0014 | 0.1286 |
| Sanger | css | no_crispr | 249 | 11 | 0.5002 | 15.4203 | 0.3785 | 0.2905 | 0.4689 |
| Sanger | added_inhibition | no_crispr | 1114 | 11 | 0.2068 | 14.9506 | 0.1873 | 0.1228 | 0.2530 |
| Sanger | zip | no_foundation | 1114 | 11 | 0.0850 | 5.3721 | 0.0852 | -0.0015 | 0.1664 |
| Sanger | bliss | no_foundation | 1114 | 11 | 0.0864 | 4.9852 | 0.0659 | -0.0133 | 0.1496 |
| Sanger | css | no_foundation | 249 | 11 | 0.4787 | 15.8611 | 0.4019 | 0.2410 | 0.5576 |
| Sanger | added_inhibition | no_foundation | 1114 | 11 | 0.2042 | 15.0404 | 0.1836 | 0.1137 | 0.2574 |
| Sanger | zip | no_targets | 1114 | 11 | 0.1147 | 5.1950 | 0.1240 | 0.0868 | 0.1661 |
| Sanger | bliss | no_targets | 1114 | 11 | 0.0464 | 4.8550 | 0.0543 | -0.0298 | 0.1376 |
| Sanger | css | no_targets | 249 | 11 | 0.3516 | 17.1108 | 0.0404 | -0.1233 | 0.1920 |
| Sanger | added_inhibition | no_targets | 1114 | 11 | 0.2172 | 14.4016 | 0.2054 | 0.1425 | 0.2663 |
| Sanger | zip | pair_mean | 1114 | 11 | 0.0541 | 5.2105 | 0.0328 | -0.0288 | 0.0850 |
| Sanger | bliss | pair_mean | 1114 | 11 | 0.0845 | 4.5854 | 0.1104 | 0.0663 | 0.1541 |
| Sanger | css | pair_mean | 249 | 11 | -0.0637 | 22.9835 | -0.0736 | -0.2078 | 0.0770 |
| Sanger | added_inhibition | pair_mean | 1114 | 11 | -0.0202 | 14.6183 | -0.0405 | -0.0984 | 0.0206 |
| Sanger | zip | ridge | 1114 | 11 | -0.0683 | 10.9977 | -0.0821 | -0.1310 | -0.0307 |
| Sanger | bliss | ridge | 1114 | 11 | 0.0425 | 11.7356 | 0.0480 | 0.0006 | 0.0951 |
| Sanger | css | ridge | 249 | 11 | -0.0098 | 33.3898 | 0.0115 | -0.1543 | 0.2141 |
| Sanger | added_inhibition | ridge | 1114 | 11 | 0.0433 | 16.9332 | 0.0133 | -0.0642 | 0.0807 |
| Sanger | zip | target_dependency_sum | 1114 | 11 | 0.1278 | 6.2320 | 0.1472 | 0.0853 | 0.2073 |
| Sanger | bliss | target_dependency_sum | 1114 | 11 | 0.1393 | 5.4880 | 0.1558 | 0.0906 | 0.2218 |
| Sanger | css | target_dependency_sum | 249 | 11 | 0.1598 | 25.7025 | 0.1560 | 0.0136 | 0.2913 |
| Sanger | added_inhibition | target_dependency_sum | 1114 | 11 | 0.1898 | 16.8752 | 0.2028 | 0.1508 | 0.2536 |

## Cross-attention gains on matched HCC lineages

A positive difference favors cross-attention. Confidence intervals resample the same held lineages for both models with common assay support.

| axis | endpoint | comparator | lineages | gain | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- |
| lco_hcc | added_inhibition | deepsynergy | 11 | 0.0051 | -0.0233 | 0.0312 |
| lco_hcc | added_inhibition | lightgbm | 11 | 0.0290 | 0.0051 | 0.0528 |
| lco_hcc | added_inhibition | matchmaker | 11 | -0.0124 | -0.0543 | 0.0201 |
| lco_hcc | bliss | deepsynergy | 11 | 0.0001 | -0.0376 | 0.0392 |
| lco_hcc | bliss | lightgbm | 11 | 0.0468 | -0.0037 | 0.0992 |
| lco_hcc | bliss | matchmaker | 11 | 0.0198 | -0.0261 | 0.0700 |
| lco_hcc | css | deepsynergy | 11 | 0.0284 | -0.0852 | 0.1243 |
| lco_hcc | css | lightgbm | 11 | 0.0187 | -0.0392 | 0.0684 |
| lco_hcc | css | matchmaker | 11 | 0.0691 | 0.0301 | 0.1134 |
| lco_hcc | zip | deepsynergy | 11 | 0.0045 | -0.0295 | 0.0390 |
| lco_hcc | zip | lightgbm | 11 | 0.0172 | -0.0074 | 0.0467 |
| lco_hcc | zip | matchmaker | 11 | 0.0070 | -0.0231 | 0.0429 |
| ldo | added_inhibition | deepsynergy | 11 | -0.0500 | -0.1141 | 0.0090 |
| ldo | added_inhibition | lightgbm | 11 | 0.1442 | 0.1006 | 0.1905 |
| ldo | added_inhibition | matchmaker | 11 | 0.0263 | -0.0390 | 0.0894 |
| ldo | bliss | deepsynergy | 11 | 0.0518 | 0.0006 | 0.0982 |
| ldo | bliss | lightgbm | 11 | 0.1405 | 0.0420 | 0.2502 |
| ldo | bliss | matchmaker | 11 | 0.0520 | 0.0008 | 0.1024 |
| ldo | css | deepsynergy | 11 | 0.1358 | -0.0479 | 0.3153 |
| ldo | css | lightgbm | 11 | 0.3007 | 0.1525 | 0.4389 |
| ldo | css | matchmaker | 11 | 0.0744 | -0.0797 | 0.2466 |
| ldo | zip | deepsynergy | 11 | 0.0245 | -0.0129 | 0.0617 |
| ldo | zip | lightgbm | 11 | 0.2172 | 0.1532 | 0.2863 |
| ldo | zip | matchmaker | 11 | 0.0624 | 0.0370 | 0.0910 |
| lpo | added_inhibition | deepsynergy | 11 | -0.0326 | -0.0569 | -0.0091 |
| lpo | added_inhibition | lightgbm | 11 | 0.0692 | 0.0295 | 0.1141 |
| lpo | added_inhibition | matchmaker | 11 | -0.0390 | -0.0786 | 0.0030 |
| lpo | bliss | deepsynergy | 11 | -0.0520 | -0.1203 | 0.0135 |
| lpo | bliss | lightgbm | 11 | 0.1295 | 0.0259 | 0.2349 |
| lpo | bliss | matchmaker | 11 | -0.0501 | -0.1227 | 0.0234 |
| lpo | css | deepsynergy | 11 | 0.0100 | -0.1722 | 0.2476 |
| lpo | css | lightgbm | 11 | 0.1433 | -0.0392 | 0.3940 |
| lpo | css | matchmaker | 11 | -0.0375 | -0.1912 | 0.1642 |
| lpo | zip | deepsynergy | 11 | 0.0390 | -0.0086 | 0.0789 |
| lpo | zip | lightgbm | 11 | 0.1500 | 0.0830 | 0.2098 |
| lpo | zip | matchmaker | 11 | -0.0172 | -0.0628 | 0.0251 |

## Joint efficacy and synergy

Positives required measured Bliss and added maximum inhibition both at least 10 percentage points.

| axis | model | joint_positives | paired_measurements | auprc | precision_top20 |
| --- | --- | --- | --- | --- | --- |
| lco_hcc | cross_attention | 22 | 1114 | 0.2659 | 0.3000 |
| lco_hcc | deepsynergy | 22 | 1114 | 0.1026 | 0.1000 |
| lco_hcc | elasticnet | 22 | 1114 | 0.1160 | 0.2000 |
| lco_hcc | lightgbm | 22 | 1114 | 0.3364 | 0.4000 |
| lco_hcc | matchmaker | 22 | 1114 | 0.1597 | 0.2500 |
| lco_hcc | mean | 22 | 1114 | 0.0273 | 0.0000 |
| lco_hcc | mono_bliss_independent | 12 | 353 | 0.0340 | 0.0000 |
| lco_hcc | no_crispr | 22 | 1114 | 0.3231 | 0.2500 |
| lco_hcc | no_foundation | 22 | 1114 | 0.2245 | 0.3000 |
| lco_hcc | no_targets | 22 | 1114 | 0.3056 | 0.4000 |
| lco_hcc | pair_mean | 22 | 1114 | 0.1838 | 0.3500 |
| lco_hcc | ridge | 22 | 1114 | 0.0619 | 0.1000 |
| lco_hcc | target_dependency_sum | 22 | 1114 | 0.0217 | 0.0000 |
| lpo | cross_attention | 22 | 1114 | 0.1626 | 0.2500 |
| lpo | deepsynergy | 22 | 1114 | 0.0878 | 0.1500 |
| lpo | elasticnet | 22 | 1114 | 0.0327 | 0.0000 |
| lpo | lightgbm | 22 | 1114 | 0.0610 | 0.0500 |
| lpo | matchmaker | 22 | 1114 | 0.0785 | 0.1000 |
| lpo | mean | 22 | 1114 | 0.0204 | 0.0000 |
| lpo | mono_bliss_independent | 12 | 353 | 0.0340 | 0.0000 |
| lpo | no_crispr | 22 | 1114 | 0.1767 | 0.2500 |
| lpo | no_foundation | 22 | 1114 | 0.1544 | 0.2500 |
| lpo | no_targets | 22 | 1114 | 0.0725 | 0.0000 |
| lpo | pair_mean | 22 | 1114 | 0.0204 | 0.0000 |
| lpo | ridge | 22 | 1114 | 0.0405 | 0.1000 |
| lpo | target_dependency_sum | 22 | 1114 | 0.0217 | 0.0000 |
| ldo | cross_attention | 22 | 1114 | 0.0259 | 0.0000 |
| ldo | deepsynergy | 22 | 1114 | 0.0206 | 0.0000 |
| ldo | elasticnet | 22 | 1114 | 0.0140 | 0.0000 |
| ldo | lightgbm | 22 | 1114 | 0.0209 | 0.0000 |
| ldo | matchmaker | 22 | 1114 | 0.0194 | 0.0000 |
| ldo | mean | 22 | 1114 | 0.0210 | 0.0500 |
| ldo | mono_bliss_independent | 12 | 353 | 0.0340 | 0.2500 |
| ldo | no_crispr | 22 | 1114 | 0.0251 | 0.0000 |
| ldo | no_foundation | 22 | 1114 | 0.0198 | 0.0000 |
| ldo | no_targets | 22 | 1114 | 0.0285 | 0.0000 |
| ldo | pair_mean | 22 | 1114 | 0.0210 | 0.0500 |
| ldo | ridge | 22 | 1114 | 0.0206 | 0.0500 |
| ldo | target_dependency_sum | 22 | 1114 | 0.0217 | 0.0000 |

## Virtual resistance-reversal hypotheses

The source-validation selector chose cross_attention. Of 428 drug pairs screened in four paired organoid donors, 342 met both positive prediction thresholds and 20 are listed. Only independent FDA No or Less concern partners enter the roster. The 95% donor interval describes variation of HCC forecasts. Full rank files also include the absolute source-validation error radius and the paired resistant–sensitive contrast. GSE182593 has no measured combination intervention for these pairs.

| anchor | candidate | dili_concern | organoid_donors | bliss | added_inhibition | added_inhibition_donor_low | added_inhibition_donor_high | added_inhibition_limore |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lenvatinib | Paclitaxel | Less | 4 | 9.8395 | 9.8167 | 7.0972 | 12.5363 | 5.5712 |
| Lenvatinib | Etoposide | Less | 4 | 8.1292 | 25.1904 | 22.4698 | 27.1386 | 23.3580 |
| Lenvatinib | LOSARTAN POTASSIUM | Less | 4 | 7.7042 | 22.6922 | 19.7108 | 27.0227 | 15.8318 |
| Lenvatinib | Mitoxantrone hydrochloride | Less | 4 | 7.3950 | 18.8786 | 16.2685 | 21.4888 | 15.9759 |
| Lenvatinib | glyburide | Less | 4 | 7.2978 | 16.3311 | 12.7089 | 20.3061 | 9.0840 |
| Lenvatinib | clotrimazole | Less | 4 | 6.0916 | 13.4990 | 10.2939 | 16.6227 | 9.8817 |
| Sorafenib | clotrimazole | Less | 4 | 5.9428 | 15.2736 | 14.5661 | 15.8260 | 10.1648 |
| Lenvatinib | Doxorubicin hydrochloride | Less | 4 | 5.4955 | 23.6831 | 21.1150 | 25.1398 | 22.6761 |
| Sorafenib | Chlorpheniramine maleate | No | 4 | 5.2433 | 15.3807 | 14.2419 | 16.9191 | 7.9992 |
| Lenvatinib | temozolomide | Less | 4 | 5.1372 | 12.5399 | 10.1708 | 15.4650 | 9.1810 |
| Lenvatinib | Chlorpheniramine maleate | No | 4 | 5.0424 | 11.5885 | 8.3010 | 14.8250 | 6.8785 |
| Sorafenib | HYDRALAZINE HYDROCHLORIDE | Less | 4 | 4.9869 | 11.2896 | 10.1662 | 12.0706 | 6.2497 |
| Sorafenib | BIBR 1048 | Less | 4 | 4.8844 | 13.0825 | 12.1481 | 14.2032 | 8.4367 |
| Lenvatinib | BENDROFLUMETHIAZIDE | Less | 4 | 4.8061 | 14.0790 | 11.5318 | 17.2158 | 9.5180 |
| Sorafenib | fenofibrate | Less | 4 | 4.7844 | 12.4162 | 12.1947 | 12.5960 | 8.2524 |
| Lenvatinib | PROPAFENONE HYDROCHLORIDE | Less | 4 | 4.6323 | 13.8170 | 11.3256 | 17.3520 | 9.9663 |
| Lenvatinib | pyrazinamide | Less | 4 | 4.6148 | 13.3416 | 11.5018 | 15.7841 | 7.6821 |
| Lenvatinib | Trametinib | Less | 4 | 4.4176 | 11.7358 | 6.4432 | 15.1334 | 6.7713 |
| Lenvatinib | ADEFOVIR DIPIVOXIL | Less | 4 | 4.3764 | 15.2636 | 13.2962 | 17.8488 | 10.9613 |
| Lenvatinib | bisacodyl | No | 4 | 4.2340 | 11.1258 | 8.8940 | 14.0721 | 7.4645 |

## Track 2 liver reference-label comparison

19 of 20 hypotheses had a unique exact stereochemical match to an independent Track 2 Murcko-held structure LightGBM forecast. The interval describes calibration-fit uncertainty for the probability of an FDA Most or Less concern label for the partner molecule. The combination ranks use the observed FDA categories and their frozen efficacy forecasts.

## Coverage

The held HCC added_inhibition endpoint has scoreable observations from Sanger.

The held HCC bliss endpoint has scoreable observations from Sanger.

The held HCC css endpoint has scoreable observations from Sanger.

The held HCC zip endpoint has scoreable observations from Sanger.

The Sanger CSS rows use a reconstruction from observed dose matrices.

The DrugComb native CSS subset has zero eligible adult HCC labels after oncology filtering.
