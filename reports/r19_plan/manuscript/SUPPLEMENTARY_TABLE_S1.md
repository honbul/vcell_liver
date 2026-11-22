# Supplementary Table S1. Development comparisons and limits

| Analysis | Contrast | Result | Evidence |
|---|---|---|---|
| S/L | Within-assay HEPG2 structure and donor-residual head or loss | The donor-matched reference improved RMSE by 0.01965 over the anchor. All six architectural edits had upper 99% bounds below 0.5% relative RMSE gain. Shuffling structure in the neural branch gave similar scores. | [Score](../SL_B_RESULTS.md) |
| X.B | Measured basal RNA versus identity/OOV | RMSE increased by 0.000108 and pooled R² decreased by 0.0366 percentage points with basal RNA in the 493-parent comparison. | [Score](../XB_RESULTS.md) |
| T.A | Structure-only FiLM calibration | Fit-only scalar reduced RMSE by 0.121% versus the uncalibrated model. Adding a gene-wise vehicle variance correction increased error. | [Score](../TA_RESULTS.md) |
| I5 | Cell Painting and viability features with fallback | The frozen reference was improved by 0.200% RMSE on 345 covered parents and by 0.046% on all 1,862. The FiLM view did not pass controls against the shared kernel, mask and permutations. | [Score](../I5_RESULTS.md) |
| Chemical similarity | Tahoe SNU-423 structural analogue | Six of 203 compounds had a nearest fitted ECFP Tanimoto ≥0.7. They accounted for 90.8% of positive squared-error reductions. Scaffold-sensitive intervals crossed zero against the best comparator. | [Score](../HCC_SCAFFOLD_RESULTS.md) |
| Low-dose JHH-5 | Matched 3.33333 µM donors | Shrinkage increased RMSE by 1.88% against the fitted scalar on 490 parents. The joint criterion was unmet. | [Score](../JHH_LOW_DOSE_TRANSFER_RESULTS.md) |
| Dose bridge | 1.11111 µM response with 10 µM donor | Incremental RMSE gain versus same-dose scalar was −0.013% on 467 parents. The gain versus a substituted high-dose donor was 0.080%, below the practical threshold. | [Score](../JHH_DOSE_BRIDGE_RESULTS.md) |
| JHH-7 position | Single plate, 1 µM | The correct compound at a different well index improved RMSE 3.14% versus an index-matched wrong compound on 80 parents, with 3.10% higher RMSE than zero response. | [Score](../JHH7_POSITION_ATTRIBUTION_RESULTS.md) |
| L1000 to Tahoe | SNU-423 with mismatched doses | On 94 weak structural neighbors, the low-rank map increased RMSE by 0.00338 versus the same-input scalar. | [Score](../SNU423_DONOR_RESULTS.md) |
