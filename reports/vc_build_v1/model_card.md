# V0 model card

This card describes the stage-0 checkpoint. The public, wet-lab-free follow-up uses the [V0+ protocol](V0PLUS_PROTOCOL.md).

## Registered use

Offline research prototype for prioritizing candidates in the LIMORE registered panel given bulk adult HCC cell-line baseline RNA. The same request returns the raw 72-hour Activity Area prediction and an empirical development interval. Larger values indicate greater measured assay activity. Report the original source curve summary and dose-window protocol. The release supports the published 88-compound registry and explicitly declines unsupported concentrations, assay/time, chemistry, input coverage or culture format. Hepatoblastoma and uncertain-age lines are excluded from the training cohort. Basal RNA is the public cell-model profile.

The package artifact contains the fitted structure standardization, gene axis, RNA preprocessing, supported compound IDs, endpoint contract and model. The interval comes from lineage-weighted cross-fit absolute errors in development. The reported held-out coverage concerns a defined C population. An expression domain flag marks nearest-fit RNA distance using the frozen feature transform. Query RNA can be supplied in any gene order. A sufficiently complete gene panel and the expected scale are required.

Current release is **V0, HCC drug sensitivity prediction** in a registered research assay. State-prediction packages are evaluated developmental artifacts with separate domain contracts. Clinical screening, treatment, toxicity, free-dose and patient benefit are outside the package. New-site utility requires local separately calibrated empirical evidence.

## Fitted artifacts

G1 root

```text
/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vc_build_v1/function/function-registered.ekrmqfpZ/full
```

`v0` contains the residual MLP release. `v0_baseline` contains the strongest registered noninteraction structure/RNA ridge release. Both include a file receipt. The R18/R19 forecasts and candidate cohorts did not supply V0 function observations.

[Functional development results, support and offline benchmark](function/RESULTS.md) · [Compound and source audit](data_function/VCB00_LIMORE.md) · [Program state evaluation](state/RESULTS.md) · [Joint-state limit](JOINT_RESULTS.md)

## Input and output

The exact actual input and output samples are in the G1 run directories. Here are selected fields for inspection. Model requests use one basal RNA dictionary plus an explicit candidate pool. The response includes endpoint, version, uncertainty, abstention and data limits. The abbreviated gene dictionary shown here is insufficient for a supported prediction.

```json
{
  "request_id": "screen-1",
  "candidate_pool_id": "limore-registered",
  "input_mode": "baseline_rna_and_structure",
  "assay_id": "CellTiterGlo_72h",
  "time_h": 72,
  "cell": {
    "cell_id": "research-cell-id",
    "lineage_role": "adult_hcc",
    "culture_format": "2D",
    "expression_scale": "log2(FPKM+1)",
    "baseline_rna": {"TP53": 1.0}
  },
  "candidates": [{"compound_id": "exact ID from the registered catalogue"}]
}
```

No training or network access occurs during prediction. Use the G1 example for a complete replay. Candidate rank is among supported rows of the submitted pool. A null value and reason signal an abstention. The output does not estimate signal at a dose or independently declare a treatment hit.

## Reproduction commands

Activate the Python environment pinned in the batch files and run from the Git repository. Artifact and source paths point to G1. Set a new output path on each run.

```bash
export PYTHONPATH="$PWD/src"
python -m vc_build.prepare --config configs/vc_build_v1/contract.yaml --bundle limore --output "$NEW_DATA"
python -m vc_build.train --config configs/vc_build_v1/contract.yaml --data-dir "$NEW_DATA" --run-dir "$NEW_RUN"
python -m vc_build.predict --model "$RUN/full/v0" --request "$RUN/full/sample_request.json" --benchmark
python -m vc_build.evaluate --prediction "$RUN/full/predictions.parquet" \
  --observations "$RUN/full/labels.parquet" --split C --endpoint-direction higher \
  --compare selected_context selected_noninteraction --out "$NEW_SCORE"
```

The data prep has explicit stage arguments and immutable output directories. The outcome file is used at the score step after forecast hashes agree. Public source data, raw outcomes, weights, and detailed predictions remain on G1 under their respective terms.
