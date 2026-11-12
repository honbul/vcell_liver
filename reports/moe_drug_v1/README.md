# Conditional experts vs size-matched Dense — unseen-drug response on Tahoe-100M (0.5 µM)

Plan: `VCELL_CONDITIONAL_MOE_EXPERIMENT_PLAN_2026-11-12.md` · Frozen protocol:
`configs/moe_drug_v1/protocol.yaml` · Branch `research/conditional-moe-drug-ood-v1` from `ea11e92`.

**Outcome: terminate the candidate.** Conditional experts (C) did not beat a size-matched Dense
model (A) — +0.07 %, interval [−0.05 %, +0.19 %] — and both lost to ridge regression on the same
inputs by about 5 %. Model B was not trained because the promotion gate failed, so the routing
question is untested.

```
main_effect: not_supported   routing_effect: not_tested   hcc_effect: not_supported
batch_confound: unresolved   scope: within_Tahoe_known_cell_lines_unseen_drug_fixed_dose
```

| document | what it answers |
|---|---|
| [RESULTS](RESULTS.md) | the five required sections: data and split, A vs C, B, SNU-423, cost / limits / decision |
| [DATA_CARD](DATA_CARD.md) | how the (cell line, drug) table was built, what was excluded and why |
| [MODEL_COMPARISON](MODEL_COMPARISON.md) | parameter matching, the 12-run grid, seed confirmation, the gate, routing behaviour |
| [superseded_v0](superseded_v0/README.md) | a cross-split identical-input leak found mid-study, and the v0 numbers it invalidated |

| table | contents |
|---|---|
| `trials.tsv` | all 16 runs with validation metrics, parameters, cost |
| `seed_results.tsv` | the selected setting at seeds 42 / 43 / 44 |
| `test_per_seed.tsv` | the single test pass, per model per seed |
| `test_per_line.tsv` | per cell line, every model and baseline |
| `per_drug.tsv` | per held-out drug |
| `plate_diagnostic.tsv` | the 239 `(line, plate)` blocks of the fixed plate read-out |
| `routing_usage.tsv` | C's expert weights, entropy and across-input variation |
| `resources.tsv` | parameters, time, inference latency, peak allocated memory |
| `selection.json` | what validation chose and how the promotion gate scored |
| `test_results.json` | test metrics, paired bootstraps, checkpoint hashes |
| `checks.json` | the four pre-run checks |
| `data_description.json` | structure similarity, truth magnitude, plate block counts |

Re-run: `notebooks/moe_drug_v1/01_reproduce_completed_experiment.ipynb`. Large arrays, checkpoints
and predictions stay on G1 at
`/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver_conditional_moe_v1`;
`manifests/moe_drug_v1/freeze.json` carries their SHA-256.
