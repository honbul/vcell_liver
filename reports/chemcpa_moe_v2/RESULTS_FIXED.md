# Conditional residual experts inside public chemCPA — corrected re-run

This replaces the chemCPA-arm results in `RESULTS.md`, which are withdrawn (see the box at its
top). The question, the data, the split, the arms' code and the pre-registered gate are unchanged;
what changed is the evaluator, and one arm was added (P2b, the router the plan specified).

**Question.** On Tahoe-100M at 0.5 µM, for drugs never seen in training but cell lines that were,
does adding conditional residual experts at public chemCPA's shared decoder hidden state predict the
mean RNA response better than the chemCPA prototype and a size-matched Dense residual in the same
place?

---

## Verdict

> In this setting — Tahoe 0.5 µM, unseen drugs in seen cell lines, chemCPA `43e830eb` with the
> authors' default hyper-parameters, residual at the decoder's last hidden state — **neither
> conditional-expert variant showed a practical gain.** Both failed every condition of the
> pre-registered promotion gate on validation, so the fixed-mixture control P3 was not trained.
> The prototype and the Dense residual did not beat the per-line mean either. Why the chemCPA
> arms trail that baseline was not established here. **This configuration is not adopted.**

| | verdict | rests on |
|---|---|---|
| `main_effect` | **negative** | validation seed mean P2 0.10436, P2b 0.10556 vs P0 0.10408, P1 0.10321 (gate failed on every condition); test seed mean P2 0.10624, P2b 0.10730 vs P0 0.10621, P1 0.10557 — P1 → P2 −0.64 % [interval includes 0], P1 → P2b −1.65 % [excludes 0] |
| `routing_effect` | **not_tested** (`not_run_by_gate`) | P3 is trained only if a routed variant passes the gate; neither did |
| `hcc_effect` | **negative** | SNU-423 test: P0 0.1138, P1 0.1133, P2 0.1138, P2b 0.1150 vs no change 0.1120, ridge 0.0982 |

One secondary goes the other way and is reported as observed: drug-centred correlation (each
line's mean response removed, per-condition correlations averaged, same rows and function for
every model) is 0.25–0.26 for the chemCPA arms against 0.097 for ridge on test. It does not favour
the routed variants.

What this does **not** say: that mixture-of-experts is unhelpful for virtual-cell models in
general. One insertion point, one expert count, two router inputs, one objective and one dataset
were tested.

---

## 1. What went wrong in the first pass, and what was fixed

The evaluator built each cell line's covariate one-hot from the manifest's `line_index` (CVCL
order) while chemCPA's loader orders it by sorted cell **name**. 45 of 50 lines — 43 of the 48
scored ones — were told they were another line; SNU-423's cells arrived labelled HEC-1-A. Training
itself used the loader's own vectors and was not affected; every validation score was, and with
it the learning-rate choice, the saved best epoch, the gate, the diagnostic and the test pass.

How large it was, measured without touching a weight: the earlier 30-epoch checkpoints re-scored
with the corrected evaluator fall from 0.276–0.296 to 0.110–0.136 on validation
(`diagnostics_e30_fixedcov.json`). The earlier report's explanation — "about two thirds of the
score is absolute-profile reconstruction error" — was describing this defect.

Fixed in the same pass, all in the scoring path (details in `RESULTS.md`, defects table):

| defect | fix |
|---|---|
| covariate one-hot by manifest order | looked up by name in the loader's `atomic_covars_dict`; check 6 compares it with the loader's own tensor per line (48/48 identical) |
| prediction on the majority plate only; 9.2 % of conditions span plates | prediction uses the same well weighting as the truth (`plate_weights.npy`; truth arrays byte-identical after the rebuild) |
| drug-centred r pooled for the arms, per-condition mean for the baselines | one definition (the baselines'), arms and baselines scored on the same rows |
| secondaries and bootstrap on the ensemble, beside the seed-mean primary | both reported, each labelled |
| router usage measured inside one condition (zero spread by construction) | spread across conditions |
| a missing seed was skipped | hard error |

Because only best-epoch checkpoints were kept, the selection could not be redone on the old runs;
the learning-rate scan and the 30-epoch runs were **re-trained** with the corrected evaluator.

## 2. Arms

| arm | residual at the decoder's last hidden state (width 256) | params |
|---|---|---:|
| P0 | none — published chemCPA | 2,249,533 |
| P1 | Dense `h + W2 GELU(W1 h)`, hidden 134, `W2` zero-init | +68,998 |
| P2 | 4 experts (rank 32), soft router on **concat(cell-line covariate embedding, drug embedding)** — as built | +68,900 (router 2,212) |
| P2b | the same experts and router on **concat(basal latent of the control cells, drug embedding)** — the router the plan pre-registered | +68,900 (router 2,212) |
| P3 | the same experts, input-independent mixture | not trained (gate) |

P2 deviates from the plan's router; it is kept under its accurate name so the evaluator fix and the
router change stay separable. P2b was added on 2026-11-13 at the user's request, before any
corrected result for either variant existed; the gate rule for two variants (each gated on its own;
P3 if either passes) was committed at the same time (`8b9c98e`). The pre-run checks, now seven,
pass for all arms including P2b (`checks.json`): identical shared initial values per seed,
bitwise-identical initial function, parameter parity 0.14 %, gradient reaching every residual
parameter, disjoint splits, covariate encoding identical to the loader's, plate weighting recorded.

## 3. Learning-rate scan (validation, seed 42, 10 epochs)

Official rates × 0.3 / 1 / 3, nothing else searched (`fixed/e10/`).

| arm | ×0.3 | ×1 | ×3 | selected |
|---|---:|---:|---:|---|
| P0 | **0.10588** | 0.10719 | 0.11485 | 0.3 |
| P1 | **0.10566** | 0.10675 | 0.11191 | 0.3 |
| P2 | **0.10692** | 0.10951 | 0.11640 | 0.3 |
| P2b | 0.11047 | **0.10894** | 0.11306 | 1.0 |

The broken evaluator had selected ×3 for every arm.

## 4. Selected setting, 30 epochs, three seeds (validation)

Same budget as the first pass's final stage, for all arms (`fixed/e30/`).

| | seed 42 | seed 43 | seed 44 | **seed mean** (primary) | 3-seed ensemble |
|---|---:|---:|---:|---:|---:|
| P0 | 0.10326 | 0.10411 | 0.10488 | **0.10408** | 0.09955 |
| P1 | 0.10370 | 0.10224 | 0.10369 | **0.10321** | 0.09816 |
| P2 | 0.10459 | 0.10495 | 0.10354 | **0.10436** | 0.09956 |
| P2b | 0.10594 | 0.10528 | 0.10545 | **0.10556** | 0.09856 |
| no change | | | | 0.10280 | |
| per-line mean (train drugs) | | | | 0.09132 | |
| ridge | | | | 0.09128 | |

Best epochs range 10–28; validation curves fall quickly for about 12 epochs and then fluctuate
within roughly ±0.003, so the runs end on a plateau rather than mid-descent.

## 5. The gate, applied as pre-registered

| condition | P2 | P2b |
|---|---|---|
| ≥ 2 % better than P0 | −0.27 % ✗ | −1.42 % ✗ |
| ≥ 2 % better than P1 | −1.11 % ✗ | −2.27 % ✗ |
| better than P1 on all 3 seeds | 1 / 3 ✗ | 0 / 3 ✗ |
| better than no-change, per-line mean and ridge | none ✗ | none ✗ |
| **promote** | **no** | **no** |

The all-seeds condition was already failed by both variants when seed 42 finished.

## 6. Test re-evaluation

**This is a bug-fix re-evaluation, not a blind test.** The 70 test drug groups were scored twice
in the first pass (with the broken evaluator), and Tahoe's test split was already a public,
project-exposed benchmark. What keeps this pass admissible is order: the runs, the selection, the
gate and the twelve checkpoint hashes were committed (`e07c0c8`,
`freeze_fixed_pretest.json`) before the job ran; the job refused to start otherwise; the hashes it
loaded match the committed ones; and nothing it produced was used to change anything. The
baselines' test predictions are the ones written in the first pass (they never went through the
broken path) and are scored here on the arms' 3,333 rows with the arms' functions.

| test, 3,333 conditions, 48 lines | seed 42 | seed 43 | seed 44 | **seed mean** (primary) | 3-seed ensemble |
|---|---:|---:|---:|---:|---:|
| P0 | 0.10658 | 0.10614 | 0.10590 | **0.10621** | 0.10230 |
| P1 | 0.10662 | 0.10457 | 0.10550 | **0.10557** | 0.10116 |
| P2 | 0.10735 | 0.10602 | 0.10535 | **0.10624** | 0.10219 |
| P2b | 0.10766 | 0.10750 | 0.10675 | **0.10730** | 0.10121 |
| no change | | | | 0.10427 | |
| per-line mean | | | | 0.09230 | |
| ridge | | | | 0.09193 | |

Paired bootstrap over drug groups (2,000 resamples), on the **primary** (seed mean of per-seed
scores); positive = the second arm is better:

| | difference | 95 % interval | relative |
|---|---:|---|---:|
| P1 → P2 | −0.00067 | [−0.00156, +0.00020] | −0.64 % |
| P1 → P2b | −0.00174 | [−0.00268, −0.00073] | −1.65 % |
| P0 → P2 | −0.00003 | [−0.00099, +0.00089] | −0.03 % |
| P0 → P2b | −0.00110 | [−0.00232, +0.00012] | −1.03 % |
| P0 → P1 | +0.00064 | [−0.00035, +0.00167] | +0.60 % |

The ensemble intervals are in `fixed/test_results.json` under their own name; they do not change
any sign that matters here (P1 → P2 −1.02 %, interval excluding zero).

Secondaries, seed mean, same rows and functions for arms and baselines:

| | drug-centred r | top-20 direction | SNU-423 macro-ΔRMSE |
|---|---:|---:|---:|
| P0 | 0.247 | 0.833 | 0.1138 |
| P1 | 0.257 | 0.833 | 0.1133 |
| P2 | 0.248 | 0.825 | 0.1138 |
| P2b | 0.255 | 0.834 | 0.1150 |
| no change | — | — | 0.1120 |
| per-line mean | 0 by construction | 0.865 | 0.0987 |
| ridge | 0.097 | 0.866 | 0.0982 |

What the test pass shows, and nothing more:

- **Main question.** Neither routed variant is better than the Dense residual: P2 is 0.64 % worse
  with an interval that includes zero, P2b 1.65 % worse with an interval that excludes it. Neither
  is better than the prototype. → `main_effect: negative`.
- **All four chemCPA arms are worse than predicting no change** on test (0.1056–0.1073 vs 0.1043)
  and about 15 % worse than the per-line mean and ridge.
- **Drug-centred correlation is the one place the chemCPA arms lead** — 0.25–0.26 against ridge's
  0.097, the same direction and size as on validation. The routed variants do not lead the
  unrouted ones on it.
- **SNU-423.** All four arms (0.113–0.115) are worse than no change (0.112) and than the per-line
  mean and ridge (0.098). → `hcc_effect: negative`.
- The plate-centred correlation of the ensembles is 0.04–0.06 (`plate_diagnostic`), i.e. little
  agreement survives removing each (line, plate) block's own mean; recorded, not interpreted
  further.

## 7. Diagnostics — validation, the checkpoints above (post-hoc)

These were computed after selection and select nothing. Arms and baselines are scored on the same
3,305 condition rows with the same functions (`fixed/e30/val_secondaries.json`,
`fixed/e30/diagnostics.json`).

### Observed

| seed mean | macro-ΔRMSE | drug-centred r | top-20 direction | RMS of predicted Δ (observed control) | RMS of predicted Δ (model's own control) | SNU-423 |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 0.10408 | 0.238 | 0.821 | 0.0819 | 0.0309 | 0.1081 |
| P1 | 0.10321 | 0.251 | 0.822 | 0.0807 | 0.0290 | 0.1078 |
| P2 | 0.10436 | 0.237 | 0.813 | 0.0808 | 0.0330 | 0.1084 |
| P2b | 0.10556 | 0.246 | 0.817 | 0.0831 | 0.0308 | 0.1097 |
| no change | 0.10280 | — | — | 0 | | 0.1095 |
| per-line mean | 0.09132 | 0 by construction | 0.862 | 0.0517 | | 0.0967 |
| ridge | 0.09128 | 0.068 | 0.861 | 0.0524 | | 0.0966 |

RMS of the true Δ is 0.1097.

1. **Drug-specific pattern vs. absolute error point in opposite directions.** On drug-centred
   correlation every chemCPA arm (0.24–0.25) is well above ridge (0.068); on macro-ΔRMSE and
   top-20 direction the two simple baselines are better.
2. **The arms do not predict a near-zero response.** Against the observed control the predicted
   Δ has RMS ≈ 0.08, larger than the baselines' ≈ 0.05.
3. **The drug-induced part the model generates is small.** The difference between the model's
   drug and DMSO predictions for the same control cells has RMS 0.029–0.033, about 0.3 of the true
   response. Since the observed-control Δ (≈ 0.081) is that difference plus the gap between the
   model's DMSO reconstruction and the observed DMSO pool, that gap has RMS of at least ≈ 0.05.
4. **Changing the control term does not change the score** much: 0.1038–0.1049 with the model's own
   control vs 0.1031–0.1055 with the observed one.
5. **Routing is not collapsed, and does move.** Mean across-condition s.d. of the expert weights
   is 0.052–0.074 for P2 and 0.029–0.044 for P2b; the largest single weight range over conditions
   is 0.36–0.51 and 0.18–0.23. Mean entropy 1.32–1.36 of a maximum 1.386.
6. **Among the arms, every secondary difference is within ~1–5 % and changes order between seeds.**

### Possible explanations — not tested here

| hypothesis | what is consistent with it | what would be needed to establish it |
|---|---|---|
| the arms' RMSE is driven by the control-reconstruction gap rather than by the response | (3) and (4) | an error decomposition that handles the cross term, or a model whose control prediction is pinned to the observed one |
| the baselines win on RMSE partly by shrinkage | they predict a smaller Δ (≈ 0.05) | a shrunk version of an arm's prediction, scored the same way |
| chemCPA's objective (per-cell Gaussian likelihood of absolute expression) under-weights the small Δ | (3) | a comparison with a Δ-targeted objective under matched conditions |
| the basal latent carries little routing-relevant information (adversarial removal of covariate information) | P2b's weights vary less across conditions than P2's | probing the latent for state information |
| 209 training drugs with RDKit descriptors limit generalisation to unseen drugs | — | a drug-representation or data-size study |

None of these is a conclusion of this study, and none is a reason to extend it.

## 8. Costs

Training, corrected re-run: 18,760 s (10-epoch scan, 12 runs) + 48,607 s (30-epoch, 12 runs) ≈
18.7 GPU-hours, plus evaluation. Per-run times are **not** comparable as model cost: P2 and P2b ran
two to a node and P0/P1 one per node. Peak allocated memory 80.1 (P0) / 81.6 (P1) / 82.9 MB (P2,
P2b).

## 9. What else happened during the re-run

The P0/P1/P2 scans were submitted 11 s before the server checkout received the commit routing them
to `runs_fix/`; they wrote to `runs/` and replaced nine earlier 10-epoch seed-42 runs from the
broken pass. Their scores remain in `trials_e10.tsv`; their checkpoints are gone. The new runs
were moved unchanged. `train_cpa.py` now refuses an existing run directory.

The primary metric samples up to 64 control cells per (line, plate) with a seeded generator; the
training-time evaluator uses the run seed and the diagnostic and test use seed 0, which moves a
score by about 0.1 % (e.g. P0 seed 42: 0.10326 vs 0.10315). The test pass uses seed 0 for every arm.

## Files

`fixed/e10/` (scan) · `fixed/e30/selection.json`, `trials.tsv` (selection, gate) ·
`fixed/e30/diagnostics.json`, `val_secondaries.json` (post-hoc) · `fixed/test_results.json` (test)
· `manifests/chemcpa_moe_v2/freeze_fixed_pretest.json` (checkpoint hashes committed before test) ·
`checks.json` · `diagnostics_e30_fixedcov.json` (size of the defect).
