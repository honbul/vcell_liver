# Vinorelbine source-screen and HCC PDX transfer

**The preregistered PDX gate failed in both publications.** The mechanism kernel improved the point estimate over the full kernel on the common six PDX patients. The paired intervals included zero in both publications. In the 2D assay and lineage holdouts, higher development scores did not reliably select the better PDX model. [Preregistered protocol](VINORELBINE_TRANSFER_PROTOCOL.md) · [Paired external results](vinorelbine_transfer_score.json)

## Public 2D training and validation

| Native AUC screen | Fit cells | Adult HCC cells | RNA/path Ridge | Full kernel | Mechanism kernel |
|---|---:|---:|---:|---:|---:|
| GDSC1 | 643 | 13 | .286 | .277 | .365 |
| GDSC2 | 648 | 12 | .422 | .441 | .419 |
| PRISM20Q2 | 462 | 14 | .490 | .475 | .496 |

Scores are Spearman on lineage groups withheld from all sources and an assay withheld entirely from that fit. The displayed penalties were selected across these same development folds. The rows sum to 1,753 cell–screen observations. The GDSC concentrations were 0.00025–0.064 µM and 0.0001–0.1 µM. PRISM used the restored HTS002 record. No PDX labels were used for settings. [Frozen development and sources](vinorelbine_transfer_forecast_receipt.json) · [GDSC native AUC definition](https://depmap.sanger.ac.uk/documentation/datasets/drug-sensitivity/)

Measured Spearman on common cells was .379 across GDSC1–GDSC2 (602 cells), .439 across GDSC1–PRISM (316 cells), and .343 across GDSC2–PRISM (307 cells). In the adult HCC subsets the estimates were .420, .595 and .214 on 12, 8 and 8 matched cells. These are descriptive cross-assay comparisons.

## Frozen PDX external technical evaluation

| Method · CPM | 2023 Spearman | 2023 Pearson | 2024 Spearman | 2024 Pearson |
|---|---:|---:|---:|---:|
| Drug mean | .000 | .000 | .000 | .000 |
| RNA and pathways Ridge | .143 | .577 | .143 | .656 |
| RNA-only kernel | −.029 | −.017 | −.086 | .175 |
| Full pooled kernel | −.029 | .410 | −.086 | .468 |
| **Pooled mechanism kernel** | **.200** | **.434** | **.143** | **.572** |
| GDSC2-only full kernel | .257 | .242 | .314 | .439 |
| PRISM-only full kernel | −.657 | −.539 | −.771 | −.522 |

The mechanism kernel's Spearman difference versus the full model was +.229 [95% −.485, +1.091] for 2023 and +.229 [−.727, +1.091] for 2024. Versus Ridge it was +.057 [−.485, +.600] and .000 [−.727, +.727]. The common group contains the same six patients in both publications. On all matched patients separately (7 and 10), the mechanism kernel reached .286 and .200. In the predeclared length-correction sensitivity on the common six, Ridge reached .486 and .600 while the mechanism kernel reached .429 in both. Raw count correction used a public approximate annotation. Table entries are estimates on preclinical PDX endpoints. The six-patient bootstrap intervals are wide. [Matched RNA and study exposure](SHUEN_PDX_RESULTS.md) · [2023 study](https://doi.org/10.3390/ijms25010017) · [2024 study](https://doi.org/10.3390/ijms25031563)

As an observation, the exact PRISM single-source model inverted the PDX ordering in both publications, while GDSC2-only gave positive point estimates. A difference in assay conditions and biological domain between the sources is a plausible contributor. The small cohort and the single compound prevent identifying which difference caused the inversion.

## Reproduction

The protocol was committed as `93c419a`. Attempt `33722` stopped at the old extension-checksum guard. Attempt `33724` stopped in development because the native COSMIC join compared decimal-serialized and integer-serialized IDs. The correction normalized the accession representation and required all three assays before fitting. Under `3ab4f87`, GPU job `33726` froze 238 values and CPU job `33727` scored separately. Both completed with exit code 0. The freeze hash is `7479c25c8dec362be5db728f6814ccfd6084e0cdc88770cee2a092e34608a85e`. The score hash is `1cc25f81baccecbfe2dfe0739b52845d52e8ff2daa694cca275671dc8723b2cd`. [Ledger](RUN_LEDGER.tsv)
