# Worker 1 HCC efficacy and local calibration

Use the isolated branch `research/confirm-hcc-efficacy-v1` based on `a255c0e8a4b4eb0a09506e9aa7d164a5d9499a99`. The G1 output namespace is `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/confirm_20261123/worker1_eff/`. Input paths are read-only. Use the corrected `source-core-92aa482/core` bundle and the original LIMORE query and detached Activity Area labels. The joint numerical contract is in [contract.json](contract.json).

## Components and job order

`run.sbatch` accepts the staged code directory followed by Python arguments. Stage `src/confirm_hcc`, `src/v0plus_v2`, and `src/vc_build` under the code directory's `src`. Include `run.sbatch`, `pdo_array.sbatch`, and `contract.json` under `code/jobs`.

| Stage | Python module | Role |
|---|---|---|
| Task 1 | `confirm_hcc.base_models` | Independently fit linear baselines, upstream CellHit, structured kernel and Track 6 ElasticNet on folds 0–2 and the complete corrected fit designated fold 3. |
| Task 1 | `confirm_hcc.pairwise component` | Refit the selected linear/tree and neural component configurations in two inner splits, the outer development split and the complete source for each historical settings member. |
| Task 1 | `confirm_hcc.pairwise member` | Learn corrected OOF stacking weights and ranking offsets at the previously selected amplitudes. Freeze outer predictions and save the corrected full-fit member. |
| Task 1 | `confirm_hcc.ledger forecast`, then `score` | Freeze every complete 50×68 forecast before reading the detached LIMORE response file. Also score the three OOF folds on native assays. |
| Task 2 | `confirm_hcc.personalization freeze`, then `score` | Freeze mean-RNA and 100 derangement forecasts and compute interaction/diversity and the single reference decision. |
| Task 3 | `confirm_hcc.pdo_data source`, then `register` | Freeze 2D priors for all study profiles and register isolated calibration labels for each patient budget. |
| Task 3 | `confirm_hcc.fewshot predict`, `score`, `transfer` | Use `pdo_array.sbatch` for the nine cohort×budget runs. The `transfer` invocation freezes the six source/target directions on exact common drug support before scoring. |
| Task 3 | `confirm_hcc.collect` | Rehash all forecasts and aggregate all nine run tables and supported panels. |
| Task 4 | `confirm_hcc.package pack`, then `replay` | Package the exact corrected structured fit, then replay the original CLI with CPU-only resources and invalid-input probes. |

Use `squeue` before submission. Worker jobs use only the `conf-eff-*` prefix. The neural array requests one GPU per element with `--array=0-2%2`. All other worker stages run on CPU. Select module options with `--help`. Use new output directories for intentional reruns. The used job IDs and output checksums are recorded in [the results directory](../../reports/confirm_hcc_efficacy_v1).

## Provenance of reused code and model settings

- `src/v0plus_v2` is the subset of the original V0+ v2 implementation at `9ed87a1a50e301dd4c90078e3d1bd32103f27468` needed to re-estimate the Pairwise recipes. The selected drug-level model settings and stack amplitudes are hash registered in `confirm_hcc.contract`. The original training table contributes group identifiers only to preserve the correspondence of the historical outer split and its settings. All fitted weights, including complete 2D models and stack weights, are new.
- The CellHit adapter fits the original upstream public `EnsembleXGBoost` class pinned by SHA-256, separately for each assay/drug. The persisted ensemble contains the original booster objects and makes the same equal-weight prediction.
- ECFP4 descriptors and pathway scoring follow Track 6 at `5929cadcd696484c5738e542f8aaa038b73c0a63`. The two ElasticNet alpha candidates are selected by a deterministic fit-only lineage split using the corrected table. The evaluation uses the common LIMORE score contract.
- To render documents from the rehashed aggregate tables, run `python3 experiments/confirm_hcc_efficacy_v1/render_reports.py reports/confirm_hcc_efficacy_v1`.
