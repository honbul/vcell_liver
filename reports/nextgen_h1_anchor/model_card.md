# H1-Anchor-Best model card

## Task and input

H1 is a retrospective assay-assisted screening procedure for a new cell lineage or organoid donor. It uses measurements from fixed assay anchors to rank other registered compounds. Larger output indicates greater predicted assay sensitivity. Each reference atlas uses the same assay where possible. Tier 4 uses a separate LIMORE reference atlas and exactly matched LICOB compounds. H1 estimates a specified laboratory endpoint. The evidence covers previously exposed research cohorts and supports decisions about further benchmarks.

Use the exact anchor identities and order from a fold package. The supported budgets are K=0, 3, 5 and 8. All four budgets return the same Q, the full registered panel minus all eight reserved anchors. A K=5 assay therefore leaves three reserved compounds outside the predicted Q. When multiple organoids represent a donor, valid inputs use the same donor-level aggregation as the reference.

## Frozen recipe and deployment package

H1-Anchor-Best is a single nested model-selection procedure. The frozen estimator, scale, ridge and A8 can differ across outer folds. Each package contains a JSON freeze, signed numerical CPU state and fit-only uncertainty calibration. The query IDs, fit groups, source hashes and configuration are recorded in the [inner freeze summary](inner_freeze_summary.json) and [anchor policy table](H1_ANCHOR_POLICIES.tsv). Individual observations and forecast files stay under the H1 G1 storage root. The repository contains the CLI and aggregate evidence.

The main arm's sample-level inputs are the assay anchors. B1 refits the existing separable structure kernel and its observed-entry solver with fit-standardized ranked RNA. This H1 RNA-only comparator uses the sequence-expression component of the corrected kernel and the registered drug-kernel weights. The secondary RNA arm promotes a dual model only when the corresponding inner fold gate passes. It is a separate evaluation arm whose data requirements differ from the standalone anchor CLI. Tier 4 and Wong consume only the registered target compound identities and target anchor measurements. Transfer rescales source Activity Area from the source fit range and uses target 1−AUC as the input orientation. The target contributes no reference donors or fitted normalization.

## CPU CLI

```bash
PYTHONPATH=src python -m nextgen_h1_anchor.cli \
  --model /mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_h123/h1_anchor/deploy-v1/LICOB-0 \
  --anchors anchors.tsv --k 5 --out predictions.tsv
```

The input has two TSV columns, `compound_id` and `response`, with one row for every fixed anchor in the chosen budget. LICOB local models expect negative native S6 AUC. LIMORE models expect native Activity Area. Pan-cancer models expect negative native fitted AUC. Tier 4 expects 1−AUC. Use a header-only file for K=0. The CLI verifies the model hash and requires finite measurements inside the frozen envelope. It returns compound IDs, scores, two interval endpoints and rank.

Intervals use the 90th percentile absolute error across inner validation predictions, falling back to the pooled estimate for drugs with fewer than 12 valid errors. They characterize development residual variation. Transfer intervals are source-variation summaries. Technical repeat variability and target assay calibration require independent measurement. All held Wong anchor packets were outside the CLI input envelope. Wong remains a descriptive transfer result. In Tier 4, 37 of 38 LICOB donors had a K=5 packet inside the deployment envelope. All 38 appear in that tier's locked retrospective score. [Budget-specific input support](cli_input_support.json) The [CLI replay receipt](cli_replay_receipt.json) checks CPU TSV results, single–batch concordance, and rejection of missing and out-of-range inputs.

## Evaluation limits

- The three pan-cancer screens have separate dense panels. Pooling their ranking scores is a macro average across endpoints and panels. The independent-lineage bootstrap couples draws across screens when they share the same group ID.
- Sparse pan-cancer response rows remain missing. Every arm uses the same complete-A8 groups and observed Q entries within a fold. This attrition is material for PRISM. Technical results are conditional on that assay support.
- LIMORE, LICOB and Wong had prior evaluation exposure. LICOB has only 38 independent donors, and the same donors appear in the distinct Tier 3 and Tier 4 analyses. LICOB transfer retains the same drug and response support across valid arms.
- Biologically plausible improvement in a retrospective ranking is a development result. The model card and results report the observed evidence and assays used for that result.


## Historical B3 and the corrected implementation

The signed outer evaluation used the frozen B3 recipes with an unpenalized anchor-mean shift and a ridge-shrunk positive slope. New recipes explicitly record a penalty of five pseudo-anchors and use `K/(K+5)` times the mean anchor residual. The five-pseudo-anchor penalty is fixed at the primary anchor budget. Historical recipes without the penalty field retain their stored behavior. The fold-specific base choice and positive slope formula are preserved.

On all 51 inner folds, the corrected recipe and the archived recipe had identical NDCG@5, Precision@5 and within-sample Spearman. The forecast audit reproduced the archived B3 values and verified identical Q ranks and ties for all 1,868 eligible observations per budget using anchor packets and forecasts alone. This establishes the rank-metric equivalence on any observed subset of Q. The archived outer MAE, RMSE and within-drug correlations remain estimates for the originally scored B3. The corrected code also clears near-zero numerical residuals when an interaction projection is degenerate. [Inner correction](H1_B3_INNER_CORRECTION.tsv) · [Forecast rank audit](b3_forecast_rank_receipt.json)

The study-level score claim protects the completed pass across output directories and records its original receipt. [Single-pass claim](outer_score_claim_receipt.json)
