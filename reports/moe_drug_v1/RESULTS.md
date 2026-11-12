# Do conditional experts predict unseen-drug responses better than a size-matched Dense model?

**Answer: no, and neither model beats a linear fit on the same inputs.**

Tahoe-100M, 0.5 µM, 48 cell lines, 70 drugs held out of every training cell line. Branch
`research/conditional-moe-drug-ood-v1`, from `ea11e92`. Companion documents: `DATA_CARD.md`
(how the table was built), `MODEL_COMPARISON.md` (how the models were matched and what validation
decided), `superseded_v0/` (a leak found and fixed mid-study).

```
main_effect:    not_supported
routing_effect: not_tested
hcc_effect:     not_supported
batch_confound: unresolved
scope:          within_Tahoe_known_cell_lines_unseen_drug_fixed_dose
```

---

## 1. Data and split — what went in, what was predicted

One training row is one **(cell line, drug) condition**, not one cell.

| | |
|---|---|
| input | that line's pooled DMSO RNA profile (2,000 genes, z-scored) + the drug's Morgan fingerprint (radius 2, 2,048 bits) |
| truth | `log1p(mean CP10k treated) − log1p(mean CP10k same-plate same-line DMSO pool)`, replicate wells equally weighted |
| never input | treated RNA, cell-line ID token, plate or well ID, treated cell counts |

349 drug groups split 209 / 70 / 70 by canonical structure, seed 20261112, stratified by plate. A
test drug is held out **everywhere**: all 48 lines, all replicate wells, and its 0.05 µM and 5 µM
rows. 16,521 of 17,450 possible conditions pass QC (94.7 %); the test split has 3,333 conditions.
The 2,000-gene panel was chosen from train responses only.

Two cell lines (NCI-H2122, NCI-H596) were dropped: median 21 and 13 cells per treated well, failing
the pre-registered cell-count rules on 0 of 398 wells. SNU-423, the pre-specified HCC line, has
207 / 69 / 70 valid drugs, so its training responses are deliberately *inside* training — this is
not an HCC zero-shot and not a leave-one-cell-line-out design.

**A leak was found and fixed mid-study.** The first build grouped drugs by stereo-aware SMILES and
split Canagliflozin — recorded in Tahoe both with and without stereocentres — across train and
validation. A radius-2 Morgan fingerprint is not chirality-aware, so the two records give the
identical input vector. Grouping now uses the stereochemistry-free parent, `checks.py` fails
outright on any identical cross-split fingerprint, and the entire validation path was rebuilt from
scratch. Test had no such pair and was never opened before the fix. Details and the v0 numbers:
`superseded_v0/README.md`.

All four required pre-run checks pass: 0 drug names, drug groups or treated wells in two splits,
0 identical fingerprints across splits, 905 other-dose wells of test drugs kept out of training,
inputs aligned with truth, both models able to fit a 64-condition batch with finite gradients, and
C's router plus all four experts receiving updates without collapsing.

---

## 2. A versus C — is there a real improvement?

**No.** The difference is a tenth of a percent with a confidence interval straddling zero.

Test macro ΔRMSE (drug mean inside a line → unweighted line mean → seed mean over 42/43/44):

| | parameters / active | test ΔRMSE | vs A | drug-centred r | top-20 direction | top-20 recall | SNU-423 ΔRMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-change | 0 | 0.12996 | −9.06 % | undefined | 0.000 ‡ | 0.006 | 0.12886 |
| per-line mean | 0 | 0.11353 | **+4.73 %** | 0.0001 † | 0.893 | 0.272 | 0.11180 |
| ridge (λ = 1) | 8,098,000 / 8,098,000 | **0.11334** | **+4.89 %** | **0.0926** | **0.895** | 0.263 | **0.11150** |
| **A** Dense (lr 3e-3, wd 1e-2) | 1,273,186 / 1,273,186 | 0.11916 | baseline | 0.0309 | 0.856 | 0.223 | 0.11602 |
| **C** Conditional experts | 1,273,076 / 1,273,076 | 0.11907 | +0.07 % | 0.0380 | 0.858 | 0.224 | 0.11606 |
| **B** Fixed experts | — | `not_run` | — | — | — | — | — |

The "vs A" column is `(A − model) / A`; positive means the model beats A. The A and C rows are
the **seed mean of the score**, which is the pre-registered primary. `test_results.json` also
carries the score of the seed-*averaged prediction* — 0.11846 for A and 0.11835 for C — which is
slightly better, as a three-model ensemble usually is. That ensemble was not pre-registered, it
does not change any ordering, and it is not used anywhere in this report. Ridge is by far the
largest model in the table at 8.1 M parameters, which is the point: capacity is not what is
missing. † the per-line mean predicts one vector for every drug, so its drug-centred prediction is
numerically zero — that 0.0001 is float noise, not a correlation. ‡ a constant-zero prediction has
no sign, so its direction agreement is 0 by construction.

Paired bootstrap over drug groups, 2,000 resamples, each drug carrying all 48 of its cell lines:

Relative values below are expressed against the **first-named** row, so "ridge − A, −5.14 %"
reads as *A is 5.14 % worse than ridge*.

| comparison | ΔRMSE difference | 95 % interval | relative |
|---|---:|---|---:|
| **A − C** | **+0.000087** | **[−0.000058, +0.000220]** | **+0.07 % [−0.05 %, +0.19 %]** |
| ridge − A | −0.005825 | [−0.007450, −0.004228] | −5.14 % |
| ridge − C | −0.005737 | [−0.007338, −0.004161] | −5.06 % |
| per-line mean − C | −0.005540 | [−0.007062, −0.004044] | −4.88 % |
| no-change − C | +0.010886 | [+0.008145, +0.013582] | +8.38 % |

The A-versus-C interval contains zero. Every model-versus-baseline interval does not. On the
seed-averaged predictions C wins in 38 of 48 individual cell lines, which sounds like something
until you see that ridge wins in **48 of 48** against C, and that C's margin over A is smaller
than either model's spread across seeds.

Validation told the same story before test was opened: C over A was −0.064 % on average across
three seeds, with the seeds disagreeing on the sign (42 and 44 favoured A, 43 was a tie to five
decimals). Against a pre-registered 2 % threshold, four of the five promotion conditions failed.

> These validation runs had A and C starting from *different* shared weights, because the parameter-
> parity solver drew from the global RNG and only A called it. Re-run from a common start on branch
> `research/chemcpa-conditional-moe-v2`, C over A is **+0.057 %** with all three seeds agreeing
> instead of −0.064 % with the seeds disagreeing. The gate still fails: +0.057 % is a thirty-fifth
> of the 2 % threshold, B is still not trained, and the test numbers below are unaffected because
> they were produced by these same frozen checkpoints. See `reports/initfix_v2/INIT_FIX.md`.

### Why both networks lose

Every one of the twelve grid settings peaks at **epoch 0–2** and gets worse from there, identically
for A and C:

| epoch | A train loss | A validation | C train loss | C validation |
|---|---:|---:|---:|---:|
| 0 | 0.0240 | 0.11632 | 0.0234 | 0.11646 |
| 3 | 0.0108 | 0.12030 | 0.0107 | 0.12102 |
| 30 | 0.0060 | 0.12336 | 0.0058 | 0.12290 |

Training RMSE falls from 0.155 to 0.077 while validation climbs past 0.123. With 9,883 training
conditions and a 2,000-dimensional output, both networks memorise the training drugs within a few
epochs. That is why a linear map beats 1.27 M parameters: ridge has more parameters than either
network and still generalises better, because it cannot memorise in the way a deep net can.

> **Two sentences here were wrong and are corrected on branch
> `research/chemcpa-conditional-moe-v2`; see `reports/initfix_v2/INIT_FIX.md` §4.** The first
> version added "— worse than predicting no change at all" to the sentence above: it is not. The
> no-change baseline on validation is **0.12618**, so a validation score of 0.123 is better than
> it, and the test table in this same section shows the trained models at ~0.119 against a
> no-change 0.12996. The overfitting is real; it does not take either network below the trivial
> baseline. The first version also said early stopping "hands back a model that has barely left its
> initialisation". It does not. This loop evaluates *after* a full pass, so `epoch 0` is the network
> after roughly 39 optimizer steps, and the untrained network was never scored. It is scored on the
> v2 branch: **0.2598–0.2663**, against 0.1164 after one epoch. Most of the distance the model ever
> covers is covered during epoch 0; what it fails to do is improve afterwards.

Nothing was done about this. The grid, schedule, capacity, dropout and patience are the
pre-registered ones; widening the search after seeing the result is what the plan forbids. This is
**not** `budget_limited` — 16 runs consumed 0.06 of the 24 permitted GPU-hours, so the recipe
stopped early because it was overfitting, not because it ran out of compute.

---

## 3. B — was the improvement due to input-dependent selection?

**`not_run`, and therefore `routing_effect: not_tested`.** The plan trains B only if C clears A by
≥ 2 % on validation with all three seeds agreeing and both minimum baselines beaten. C cleared none
of those. Training B would have measured the source of an improvement that does not exist. The 8
runs reserved for B were not spent.

One observation about C's trained router, which is a description and **not** a substitute for B.
Across the three frozen C models the router is essentially constant and essentially uniform:
entropy 1.3807–1.3826 against a maximum of 1.3863, mean weights ≈ [0.24, 0.29, 0.23, 0.24], and a
standard deviation of each weight **across inputs** of only 0.007–0.008. The mixture barely moves
with the cell or the drug. So C converged to something close to what B is by construction — which
is consistent with C ≈ A, but it is an observation about one trained model under one recipe, not
evidence about whether conditional routing can help.

---

## 4. SNU-423 — does anything show up in the HCC-derived line?

**No.** The pre-specified secondary line reproduces the overall picture exactly.

| | SNU-423 test ΔRMSE |
|---|---:|
| no-change | 0.12886 |
| per-line mean | 0.11180 |
| ridge | **0.11150** |
| A | 0.11602 |
| C | 0.11606 |

A − C on SNU-423 alone: −0.000032, 95 % interval [−0.000171, +0.000106] — zero again, with 70 test
drugs. Ridge beats both networks here by about 4 %. SNU-423 was never allowed to select a setting,
and no setting was re-chosen after seeing it.

Since there is no general method effect, the plan's "general gain / HCC failure" split does not
arise. `hcc_effect: not_supported` means *no improvement was found to transfer*, not that something
specific to liver cells broke.

---

## 5. Cost, limits, decision

### Cost

| | runs | mean train time | inference, batch 256 | peak allocated |
|---|---:|---:|---:|---:|
| A | 8 | 10.5 s | 0.47 ms | 199 MB |
| C | 8 | 16.4 s | 0.79 ms | 199 MB |

16 runs, 214.8 s of training loop in total — **0.06 GPU-hours against a 24 GPU-hour budget**, on
one A30, one GPU per run; the longest single run was 18.9 s against a 1 GPU-hour per-run cap. The
Slurm allocations holding them were a few minutes each. C costs about 1.6× A's time at matched parameter count, because all four experts are
computed; no speed claim was made and none is contradicted. `sacct` remains unreachable for this
account, so there is no scheduler-side memory figure — the same limitation the earlier studies
recorded. Peak memory here is `torch.cuda.max_memory_allocated`.

### The limit that matters most: how much drug-specific signal is in this truth at all

The fixed plate diagnostic removes each `(cell line, plate)` block's own mean from prediction and
truth over 239 blocks of ≥ 5 evaluated drugs:

| | drug-centred r | after also removing the plate-block mean |
|---|---:|---:|
| ridge | 0.0926 | **0.0096** |
| C | 0.0380 | 0.0187 |
| A | 0.0309 | 0.0158 |
| per-line mean | 0.0001 † | 0.0002 † |

Ridge's drug-centred correlation collapses by an order of magnitude once the plate block is
centred out. Most of what looked like drug specificity in the best-performing model is aligned with
which plate the drug sat on. The networks retain slightly more after centring, but from a far lower
starting point, and none of these numbers is large.

Read together with the headline table, the honest summary of this condition set is that **almost
all of the achievable error reduction is the per-cell-line average response**, and that the
drug-specific part any of these models recovers for unseen drugs is small and partly confounded
with plate structure. This diagnostic bounds the problem; it does not remove it, and no corrector
was fitted on test truth.

### Other limits

- Train and test conditions share plates and share the DMSO pools that define their truth. Holding
  out drugs does not remove a per-plate measurement environment. The earlier study measured 0.76
  correlation between untreated DMSO wells, so a large shared component exists regardless of drug.
- Tahoe is a dataset this project has analysed before. A new seed does not make it a blind cohort;
  this is a *pre-registered held-out drug evaluation within Tahoe*, and what is protected is the
  selection procedure, not the secrecy of the data.
- Drug-identity holdout, not scaffold holdout: median nearest-train Tanimoto of the test drugs is
  0.25, but 5.7 % sit above 0.7.
- One dose (0.5 µM). Nothing here speaks to dose response.
- The bootstrap resamples drug groups conditional on these trained models. It is not an interval
  over training randomness and not a claim about independent laboratories. The 48 cell lines are
  not independent biological replicates.
- Four experts at rank 32 is a single configuration the plan deliberately chose not to sweep.

### Decision

**Terminate this candidate.** Conditional experts did not improve unseen-drug response prediction
over a size-matched Dense model on this data, at this scale, under this recipe, and both models
were beaten by ridge regression on the same inputs by about 5 %.

What is *not* concluded: that mixture-of-experts architectures do not help RNA perturbation models
in general. The comparison was fair but it was a comparison between two models that both overfit
before learning anything a linear map could not, so it measures this configuration, not the idea.

No further hyperparameter search, dataset or model is being added to explain the result — that is
exactly the escalation the plan rules out. The useful next question this surfaced is a different
one and belongs to a different pre-registration: **before more capacity is worth comparing, the
truth definition needs a drug-specific signal that survives plate centring.** The plate diagnostic
above, and the DMSO-to-DMSO correlation of 0.76 recorded earlier, are where that work would start.

---

## Completed and not completed

| | |
|---|---|
| completed | data build and manifests; 4/4 pre-run checks; 3 baselines on validation and test; 12-run A/C grid at seed 42; selected setting at seeds 43 and 44; selection and promotion gate; freeze; one test pass with one code path for every model; paired bootstrap; plate and per-line diagnostics |
| not completed, by rule | model B (gate failed — `routing_effect: not_tested`) |
| not completed, blocked | scheduler-side peak memory (`sacct` unreachable) |
| runs used | 16 of 24 permitted; 0.06 of 24 GPU-hours of training |

Artifacts in Git: `configs/moe_drug_v1/protocol.yaml`, `manifests/moe_drug_v1/{sources,splits,
drug_aliases,genes,coverage,transform,freeze}.*`, `reports/moe_drug_v1/{trials,seed_results,
test_per_seed,test_per_line,per_drug,resources,routing_usage,plate_diagnostic}.tsv`,
`reports/moe_drug_v1/{checks,selection,test_results,data_description}.json`.
Large arrays, checkpoints and predictions stay on G1 under
`/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver_conditional_moe_v1`; `freeze.json` carries
their SHA-256. Re-run commands: `notebooks/moe_drug_v1/01_reproduce_completed_experiment.ipynb`.
