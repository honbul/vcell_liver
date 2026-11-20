# Two initialisation defects in the v1 comparison, and what changed when they were fixed

Branch `research/chemcpa-conditional-moe-v2` from `c667041`. The v1 branch, its runs and its reports
are untouched; this is a separate record, not a rewrite of one.

Scope: the two defects named in the goal, the six validation re-runs of the **already-selected**
setting, and two incorrect sentences in the v1 report. No new hyper-parameter search, no model B,
no test pass.

---

## 1. Defect (a) — counting parameters changed what the models were initialised to

`solve_dense_hidden` fixed A's bottleneck width by building trial models and counting their
parameters. It built **197** of them, each drawing from the global RNG. Only kind A called it. So
after the common `torch.manual_seed(seed)` at the top of a run:

```
A run:  seed -> 197 throwaway models -> shared encoders / trunk / decoder
C run:  seed ->                         shared encoders / trunk / decoder
```

The two kinds' *shared* modules — the parts the comparison holds fixed — came out of different RNG
states. Measured on the trunk at seed 42, A's and C's shared weights differed by **91 %** of their
own mean magnitude: not a perturbation, two independent draws.

| seed | v1 A shared SHA-256 | v1 C shared SHA-256 | identical |
|---|---|---|---|
| 42 | `598760a270fc509a` | `769302e35542f0e7` | no |
| 43 | `eadc567e75f3688f` | `2f2193ed03a8e0b4` | no |
| 44 | `0722381030c2eeba` | `54d0d19be5b9b7fd` | no |

**Fix.** Two separate changes, because the first alone is not enough. The parameter scan now runs
inside `torch.random.fork_rng`, so counting cannot move the global stream. And the shared modules
are gathered into one `SharedCore` whose initial values are produced once per seed by
`shared_init_state` and **copied** into each model by `build(..., seed=...)` — so the shared weights
no longer depend on the draw order of the head that follows them. Every run records
`shared_fingerprint()`, and `checks.py` now fails outright if A, B and C disagree at seeds 42/43/44.

| seed | A | B | C | identical |
|---|---|---|---|---|
| 42 | `cb86133d57cbb814` | `cb86133d57cbb814` | `cb86133d57cbb814` | yes |
| 43 | `b7dff3ea2de4308c` | `b7dff3ea2de4308c` | `b7dff3ea2de4308c` | yes |
| 44 | `daaf6ffd48038342` | `daaf6ffd48038342` | `daaf6ffd48038342` | yes |

The six re-runs below carry these fingerprints in `runs.tsv`, and A and C match at every seed.

## 2. Defect (b) — the experts' two layers had their initialisation bounds swapped

`ExpertBank` stores `w1` with shape `[K, d, r]` and contracts it `bd,kdr->bkr`, so that map's fan-in
is `d = 256`. `nn.init.kaiming_uniform_` reads fan-in off dimension 1 of a 2-D slice, which is
`r = 32`. The second layer has the mirror problem. The bounds were therefore exchanged:

| | fan-in | v1 sd | correct sd | v1 / correct |
|---|---:|---:|---:|---:|
| `w1` (d → r) | 256 | 0.1016 | 0.0361 | **2.82×** |
| `w2` (r → d) | 32 | 0.0360 | 0.1021 | **0.35×** |

Each expert is meant to be exactly `Linear(256,32) → GELU → Linear(32,256)`; PyTorch's default for
those layers is `U(-1/√fan_in, +1/√fan_in)`. The bounds are now written out from the contraction
that actually happens rather than inferred from the storage layout, and `checks.py` compares the
realised standard deviations against `1/√(3·fan_in)`.

The net effect on the expert bank's *output* scale at initialisation was small (0.247 → 0.223), so
this is a correctness fix rather than an explanation of the v1 result. The imbalance was between
the layers, not in the total.

`src/moe_drug_v1/init_audit.py` reproduces the v1 code and measures both defects; its output is
`init_audit.json`.

---

## 3. The six re-runs

Same condition table as v1, byte for byte: same truth definition, same 209/70/70 drug split, same
48 cell lines, 9,883 training conditions. Same already-selected setting, lr 3e-3 / wd 1e-2. Only the
initialisation changed. Validation only.

| kind | seed | best epoch | epochs | val at init | val macro ΔRMSE | s |
|---|---:|---:|---:|---:|---:|---:|
| A | 42 | 0 | 31 | 0.26112 | 0.116448 | 12.8 |
| A | 43 | 0 | 31 | 0.26626 | 0.116418 | 10.2 |
| A | 44 | 0 | 31 | 0.26547 | 0.116451 | 13.0 |
| C | 42 | 0 | 31 | 0.25982 | 0.116385 | 16.2 |
| C | 43 | 0 | 31 | 0.26410 | 0.116310 | 16.5 |
| C | 44 | 0 | 31 | 0.26516 | 0.116425 | 14.8 |

### The two aggregates, which are not the same number

| | A | C | C over A |
|---|---:|---:|---:|
| seed mean of the scores (**pre-registered primary**) | 0.116439 | 0.116373 | **+0.057 %** |
| score of the seed-averaged prediction (3-model ensemble, not pre-registered) | 0.115729 | 0.115676 | +0.046 % |

Per seed, C over A: **+0.054 %, +0.093 %, +0.022 %**. All three now agree in sign.

`validation_summary.json` writes these under `seed_mean_of_scores` and
`score_of_mean_prediction` and never merges them. The ensemble is better than any single seed for
both models, which is what averaging three predictions usually does; it is a different quantity, not
a better estimate of the same one.

### What this changes, and what it does not

v1 reported C over A as **−0.064 %** with the seeds disagreeing in sign. With a common start it is
**+0.057 %** with all three agreeing. The sign flipped and the seed-to-seed noise collapsed
(sd 5.9e-05 for C, 1.9e-05 for A) — which is the expected consequence of removing an uncontrolled
difference between the arms, and is the reason the defect mattered.

It does not change the decision. The pre-registered promotion threshold is **2 %** relative
improvement on validation. +0.057 % is about one thirty-fifth of it. Model B is still not trained,
the routing question is still `not_tested`, and both models still lose to ridge on the same inputs.
`main_effect: not_supported` stands.

Note on what "all three seeds agree" buys: three runs that differ only in seed are not three
independent measurements of a population, and a consistent sign across them is not a significance
statement. It is reported because the pre-registered gate asks for it, and the gate fails on
magnitude regardless.

---

## 4. Two sentences in the v1 report that were wrong

**"early stopping hands back a model that has barely left its initialisation"** — `RESULTS.md` §2.
It had not. The loop evaluates after a full pass, so `epoch 0` in `history.tsv` is the network
*after* ~39 optimizer steps, not the untrained one. The untrained network was never scored, so there
was nothing in the record to support the claim. It is scored now, as `val_at_init`: **0.2598–0.2663**
against 0.1164 after one epoch. The model covers most of the distance it ever covers during epoch 0;
what it does not do is improve after it.

**"validation climbs past 0.123 — worse than predicting no change at all"** — `RESULTS.md` §2. The
no-change baseline on validation is **0.12618**. A validation score of 0.123 is better than it, not
worse. The overfitting is real — training RMSE falls to 0.077 while validation rises — but it does
not take either network below the trivial baseline on validation. (On *test*, where no-change is
0.12996, the trained models at ~0.119 also beat it; the v1 test table already showed this, which is
how the sentence contradicted its own tables.)

Both sentences are corrected in `reports/moe_drug_v1/RESULTS.md` on this branch, with the v1 text
quoted so the change is visible.

---

## 5. Files

| | |
|---|---|
| `runs.tsv` | the six re-runs with their shared-init fingerprints |
| `validation_summary.json` | the two aggregates, separately named |
| `init_audit.json` | both defects measured on the v1 code |
| `checks.json` | the pre-run checks, now including shared-init identity |
| code | `src/moe_drug_v1/{models,train,checks,init_audit,initfix_report}.py` |
| runs | `experiments/initfix_v2/{env.sh,checks.sbatch,rerun.sbatch}` |

Re-run: `sbatch experiments/initfix_v2/checks.sbatch` then
`sbatch experiments/initfix_v2/rerun.sbatch`, then `src.moe_drug_v1.initfix_report`.
Outputs live on G1 under
`/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver_chemcpa_moe_v2/initfix_v2`.
