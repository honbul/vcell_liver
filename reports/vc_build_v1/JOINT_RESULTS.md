# Public state-supervision comparison

Slurm 32310 froze forecasts and 32311 independently scored the frozen artifact on 6,776 LIMORE cells-by-compounds across 77 HCC models and 74 independent family groups. The three variants share the same five functional folds and input budget. Each fold has one or two adult HCC state-supervision sources, six technical nonhepatic contexts and 944 jointly measured basal genes.

| Model | Cell macro MAE | NDCG@5 | Paired lineage-macro MAE gain vs Direct, 95% CI |
|---|---:|---:|---:|
| Direct | 0.10969 | 0.89274 | Reference |
| Multi-output | 0.10943 | 0.89208 | +0.17% [−0.39%, +0.71%] |
| State-assisted | 0.10820 | 0.89828 | +1.33% [+0.18%, +2.51%] |

State-assisted meets the directional error condition against the matched Direct. It falls below the 5% practical target. Its gain against Multi-output is +1.17% [+0.03%, +2.47%]. The NDCG@5 interval against Direct includes zero. The strongest standalone LIMORE context model has lower absolute MAE under the separately registered functional protocol.

State supervision used distinct public assays. It contains sparse matching adult contexts and no matched local sister-culture measurements. The comparison provides development evidence for a small auxiliary learning effect. State output eligibility and prospective top-five utility require their own measurements and criteria.

[Aggregate paired metrics](joint/metrics.json), [fold support and source hashes](joint/support.json), [freeze receipt](joint/predictions.parquet.freeze.json).
