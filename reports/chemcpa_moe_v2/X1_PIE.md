# X1 — PIE (chemical / Tahoe) as an external reference point

Verdict up front, because the two usual reasons do **not** apply here and the real one is narrower:

```
output_contract:   MATCHABLE     PIE's label is mean(log1p(CP10k)) just like v2's; the only
                                 difference is a per-cell-line control where v2 uses a
                                 per-plate one, and that difference is a computable offset
holdout:           VIOLATED      the only public model that has seen chemical data was trained
                                 on Tahoe at weight 0.68, so it has seen every one of this
                                 study's 70 held-out test drugs
X1 status:         not_run       retraining `tahoe_wdataset` on this study's split is a
                                 separate study-sized job, not a comparator inside this budget
```

Everything below is read off the pinned repositories, not inferred.
Sources: `github.com/ArcInstitute/pie` at `0389400`, `huggingface.co/api/models?author=arcinstitute`
(queried 2026-11-12), and the Tahoe dataset card.

## 1. The output contract is not the blocker

`pie.process.normalize` is "CP10k + natural log1p normalization of count matrices", applied **per
cell**. `pie/prep/labels.py::pseudobulk_h5ad` then takes `mat.mean(axis=0)` over the cells of a
`(context, perturbation)` group, and `pseudobulk_delta_p` writes

```python
delta_p[i] = all_pseudo[(ctx, pert)] - all_ctrl[ctx]
```

So PIE's Δ is **mean over cells of log1p(CP10k), treated minus control** — the same functional form
this study's v2 truth uses, and exactly the form v1 did *not* use. The one difference is which
control:

| | treated side | control side |
|---|---|---|
| this study (v2) | mean of per-cell log1p over the drug's valid wells | the **same plate's** DMSO cells of that cell line |
| PIE (`tahoe.yaml`) | mean of per-cell log1p over the `drug_dose` group | `DMSO_TF_0.0uM` of that **cell line**, pooled over plates |

`configs/prep/dataset/tahoe.yaml` sets `context_col: context` and `control_label: DMSO_TF_0.0uM`,
and `ctrl_means.npy` is `(C, G)` with one row per context. There is no plate axis.

Because each 0.5 µM condition's wells sit on one plate, and because both sides are means of the
same per-cell quantity, the two labels differ by a term this study already computes:

```
Delta_v2 = Delta_PIE + ( ctrl_line_mean - ctrl_plate_mean )
```

Both terms are in `assets/v2/raw` and neither uses any held-out response. So converting PIE's output
to this study's contract is arithmetic, not a blocked contract. That is why this is **not**
`output_contract_blocked`.

Two smaller frictions, recorded but not decisive: PIE's Tahoe gene axis is 18,151 genes (or 17,845
without a `genes=` file) against this study's 2,000-gene panel, and `pie prep` accepts
`genes=<file>`, so the axis can be fixed to the panel at preprocessing time.

## 2. The holdout is the blocker

Two PIE model repositories are public: `arcinstitute/PIE_replogle_wdataset` and
`arcinstitute/PIE_replogle_xdataset`. There is no `PIE_tahoe*` model repository.

`PIE_replogle_wdataset` is trained on genetic CRISPRi perturbations only — the wrong task.

`PIE_replogle_xdataset` is the only public model that has seen chemical data, and
`configs/experiment/replogle_xdataset.yaml` says what it saw:

```yaml
  preprocessed_dirs:
    - .../PIE_replogle_nadig_essential@.../preprocessed
    - .../PIE_tahoe100m@e57e281a.../preprocessed
    ...
  dataset_weights:
    replogle: 0.0
    tahoe: 0.68
```

`PIE_tahoe100m` is the whole of Tahoe-100M: 95,624,334 cells, 1,137 drug-dose conditions, 50 cell
lines. Every one of this study's 70 held-out test drug groups and all 48 cell lines are inside it,
at 0.68 of the training weight. Scoring that checkpoint on this study's test drugs would be scoring
a model on its own training data. The plan's rule — exclude the test drugs' treated responses from
training, reconstruction and retrieval evidence in every cell line — rules it out, and no
post-hoc correction recovers it.

## 3. What running X1 properly would cost

`configs/experiment/tahoe_wdataset.yaml`: `trainer.devices: 2`, `num_nodes: 1`, `max_steps: 5000`,
`accumulate_grad_batches: 4`. `AGENTS.md` gives a time only for the Replogle recipe (2 × H100 80GB,
about 1.2 h and 20 GiB per GPU per fold) and no figure for Tahoe. On top of training:

- a `pie prep` pass defining this study's split as PIE split JSONs, so the 70 test drug groups are
  held out of training *and* out of the evidence retriever;
- the `PIE_sources` knowledge directories the recipe requires — context text, perturbation text,
  SMILES, L1000 TAS, PRISM secondary, JUMP morphology — which are the model's premise and must be
  downloaded and version-pinned;
- optionally a re-prep with `genes=` fixed to the 2,000-gene panel and, if the plate-matched control
  is wanted natively rather than by the offset above, a context definition this recipe does not have.

That is a second study, not a baseline column in this one. It is recorded here as **not run**, with
the reason, rather than attempted at a scale that would not finish or claimed as impossible.

## 4. What this does to the comparison

X1 was to be an external reference point, not a term in the hypothesis. Its absence does not change
what P0 / P1 / P2 measure. What it does remove is the one planned check on whether a much larger
knowledge-conditioned model does better on this task than chemCPA does, so the study's conclusions
are about chemCPA-scale models on this data and say nothing about PIE.
