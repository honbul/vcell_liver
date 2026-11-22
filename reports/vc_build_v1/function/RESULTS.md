# VCB-00/01/03 functional result

This run produced two fit packages for the published 72-hour LIMORE CellTiter-Glo Activity Area endpoint, a separate evaluator, and a runnable offline prediction request. The release contains a 32-unit, three-seed residual MLP and the best selected noninteraction structure/RNA ridge. Their current scope is research-only curve-summary ranking of a registered 88-compound panel for adult HCC cell-line RNA. A clinical effect, actual 72-hour cell count, concentration-specific response, and matched state change are outside the endpoint.

## Held lineage development

LIMORE Table S1, GEO basal RNA, Table S4 Activity Area, and exact standardized compound identity yielded 6,776 observed responses for 77 age-confirmed adult HCC models, 74 related-lineage groups, and 88 compounds. The earlier project used LIMORE. The current outer labels were excluded by the registered fit protocol. This is public lineage-holdout development.

| Arm | C cell-macro MAE | C cell-macro NDCG@5 | C interval coverage |
|---|---:|---:|---:|
| Drug mean | 0.11154 | 0.86924 | 90.01% |
| Additive RNA ridge | 0.10982 | 0.86924 | 90.08% |
| Five basal RNA neighbors | 0.10846 | 0.89640 | 89.68% |
| Inner-selected noninteraction, structure/RNA ridge | 0.10887 | 0.86924 | 90.14% |
| Inner-selected context, residual MLP | 0.10431 | 0.90434 | 90.23% |
| Context MLP with RNA set to the fit coordinate mean | 0.11262 | 0.86683 | — |
| Context MLP given another lineage's RNA | 0.12138 | 0.86398 | — |
| Random ranking | — | 0.51875 | — |

**H1 missed the predeclared practical effect gate.** The MLP's relative lineage-macro MAE improvement against its strongest registered noninteraction baseline was **3.833%** with paired lineage 95% interval **0.456% to 6.994%**. The rule requires both at least 5% point improvement and a positive lower bound. Improvement against drug mean was 6.385% [0.801%, 11.391%]. Against the strongest noninteraction arm, the paired NDCG@5 difference was +0.03527 [0.02021, 0.05114]. NDCG and top-5 rank overlap use continuously observed activity; prospective hit and equal-budget cost remain unevaluated.

Same-drug within-fold differences between distinct measured lineages had descriptive MAE 0.14563 with the MLP, 0.15049 with the noninteraction arm and 0.15055 with drug mean (48,576 correlated cell-pair contrasts per arm). Perturbing basal RNA by replacing or removing it increased error. The differences show limited context information under this assay and do not establish external-lineage personalization or real screening savings.

## Drug and double-cold diagnostics

| Track | Common rows | Selected context MAE / NDCG@5 | Selected noninteraction MAE / NDCG@5 |
|---|---:|---:|---:|
| D | 6,776 | 0.16309 / 0.61969 | 0.16821 / 0.63374 |
| CD | 1,356 | 0.17130 / 0.62600 | 0.16552 / 0.64394 |

Median and maximum nearest fit ECFP Tanimoto for the 88 withheld chemical identities were 0.341 and 0.797. CD covered intersections of the five independently assigned cell and chemical fold labels. Any lineage-resampled D/CD intervals in the machine-readable evaluator summarize these observed chemical folds; they do not estimate independent scaffold uncertainty. The shipped LIMORE release uses the registered C panel.

## Execution and CLI

- Slurm 32303 trained the 924-observation real-data smoke. Slurm 32305 independently scored it after the forecast freeze.
- Slurm 32304 prepared 124,684 forecasts, tuned on inner lineage or chemical groups, and froze output once. Slurm 32306 scored C/D/CD separately and checked the same artifact receipt.
- G1 run root is `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vc_build_v1/function/function-registered.ekrmqfpZ`. The full forecasts and the measurement file are separate. The freeze SHA-256 is `affd6e3c8a3cd1eff7071d5342387a166579c9455be2736768c0ac6b5c4f5b5a`.
- The development quantile construction gives the released MLP a 90% nominal interval half-width of 0.246595 AA. Its held C coverage is shown above. Intervals and RNA-domain diagnostics reflect the observed development distribution.
- The CPU fit ran for 2 min 35 s wall time, at 656,312 KiB peak RSS. An eight-candidate RNA replay succeeded for eight and took 1.40 s. A 100-candidate batch on the same deposited RNA scored all 88 registered compounds and abstained on 12 deliberately unregistered IDs in 3.00 s with 222,680 KiB peak RSS. These figures are single observed offline CPU request measurements including model load. The replay used one development line; C scoring supplies the separate withheld-lineage evidence.

With the run's `repo/src` on `PYTHONPATH` on G1, invoke:

```bash
python -m vc_build.predict \
  --model "$RUN/full/v0" --request "$RUN/full/sample_request.json" --benchmark
```

The G1 package receipt, aggregate metric and input hashes, and release contract are recorded in the evidence artifact. Weights, expression, forecasts, detailed per-cell scores and replay requests remain in G1 storage. [Functional evidence](functional_evidence.json) [Registered protocol](../FUNCTIONAL_PROTOCOL.md) [LIMORE source](https://pmc.ncbi.nlm.nih.gov/articles/PMC7505724/)
