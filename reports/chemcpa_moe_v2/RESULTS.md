# Conditional residual experts inside public chemCPA — what was run and what came out

One question: on Tahoe-100M at 0.5 µM, for drugs never seen in training but cell lines that were,
does adding **conditional residual experts** to public chemCPA's shared decoder hidden
representation predict the mean RNA response better than the chemCPA prototype and than a
**size-matched Dense** residual in the same place?

Every number below was produced by a job in `experiments/chemcpa_moe_v2/` or
`experiments/initfix_v2/` on this branch. Where something was not run, it says so and why.

---

## Verdicts

| | verdict | what it rests on |
|---|---|---|
| `main_effect` | see §9 | the single test pass over 70 held-out drug groups |
| `routing_effect` | **not_tested** | the pre-registered promotion gate was not met on validation, so P3 was never trained |
| `hcc_effect` | see §9 | SNU-423 (`CVCL_0564`), the pre-specified secondary |

`routing_effect` is settled already and does not depend on the test pass: §7 gives the gate and
the four conditions, two of which failed.

---

## E0 — The two initialisation defects in the v1 code (goal §0)

**Purpose.** v1's comparison claimed A and C started from identical shared weights. They did not.
Before anything new is built on that code, measure the damage and fix it.

**Data.** Unchanged: the v1 condition table, the v1 split, the v1 truth. No new data, no new
hyper-parameter search, no new test-set selection — all four were explicitly out of scope.

**Model.** The v1 small MLPs A (shared trunk only) and C (trunk + conditional experts).

**What changed.**

1. `solve_dense_hidden` sized the Dense residual by *constructing* candidate models in a loop, and
   every construction drew from the global RNG. Only A called it. Measured: **197** throwaway
   models were built, and A's and C's trunk weights ended up differing by **91 %** of their own
   mean magnitude. Fixed by wrapping the search in `torch.random.fork_rng` and by copying one
   explicit `shared_init_state` into both models.
2. `ExpertBank` stored `w1` for `bd,kdr->bkr` but initialised it with Kaiming fan-in computed on
   the transposed shape. Measured: `w1` was **2.815×** too wide and `w2` **0.353×** too narrow
   against the correct standard deviations 0.036084 and 0.102062. Fixed by bounding each from the
   contracted dimension it is actually multiplied along.

**Result.** Six validation runs, A and C at seeds 42/43/44. The shared-initialisation hash now
matches across kinds at every seed (`shared_init_matches_across_kinds: {42,43,44} → true`).

| | A | C |
|---|---:|---:|
| seed mean of scores | 0.116439 | 0.116373 |
| seed sd | 0.000019 | 0.000059 |
| score of the mean prediction (3-model ensemble) | 0.115729 | 0.115676 |

C over A: **+0.0567 %** on the pre-registered primary, three of three seeds agreeing in sign.
v1 had reported −0.064 % with seeds disagreeing. The two aggregates are kept under separate names
everywhere because they answer different questions; they are not two estimates of one quantity.

Two sentences in the v1 reports were wrong and are corrected in place, with the original claim
quoted above each correction:

- "the model barely left its initialisation" — the recorded score at initialisation is
  0.2598–0.2663, against a trained 0.1164. It moved.
- a no-change comparison that had the inequality backwards — no-change on v1 validation is
  0.12618, so 0.123 is **better** than no change, not worse.

**Meaning.** The fix changes the sign of the v1 effect but not its size: +0.06 % is one
thirty-fifth of the 2 % the plan requires before a conditional-expert result is worth promoting.
The v1 negative result stands, now on correct code.

---

## E1 — Running public chemCPA before building on it (goal §2)

**Purpose.** Not to reproduce a published number, but to see the model's real input/output
contract and native training loop before attaching anything to it.

**Data.** The authors' own `sciplex_complete_middle_subset.h5ad`, fetched from the Backblaze URL
their repository gives. Google Drive refused `gdown`, so the Drive copy was not used.

**Model.** chemCPA at `43e830eb0958c54e4aa64442c17ec0fed19b3f15`, unmodified, with the authors'
`config/model/hparams/default.yaml` copied verbatim into
`configs/chemcpa_moe_v2/hparams_official.json`.

**Result** (`reports/chemcpa_moe_v2/chemcpa_example.json`), read off a real checkpoint:

```
input  genes            (B, 2000)
output                  (B, 4000) = [mean | softplus(variance)]
latent_basal            (B, 32)      encoder(genes)
drug_embedding          (B, 32)      dose-scaled, from the amortized doser
latent_treated          (B, 32)      latent_basal + drug + covariate
cell_drug_embedding     (B, 64)
objective               GaussianNLL on the cell's own expression, minus reg_adversary * drug
                        adversary and reg_adversary_cov * covariate adversary; every third
                        global step updates the adversaries instead, with a gradient penalty
```

Four things had to be dealt with to get it to run, all recorded because they are part of "what it
takes to use this code": the h5ad was in the pre-2021 AnnData layout and had to be re-saved once;
`split_cellcycle_ood` spells its held-out level `odd`, so `split_random` was used; the repo ships
a dangling `project_folder` symlink pointing at the authors' private path; and the authors'
embedding validation loop is O(n_cells) and does not finish on a 1.27 M-cell file.

**Meaning.** The insertion point for a residual is unambiguous: the decoder's last hidden state,
width 256, the input to its final `Linear`. Nothing in the prototype's structure, loss or output
head is removed anywhere in this study.

---

## E2 — The v2 truth, and why the shards had to be read again (goal §3)

**Purpose.** v1 aggregated a well as `log1p(mean CP10k)`. The plan fixes the common truth as
`mean(log1p(CP10k))`, treated minus the **same plate's** DMSO pool. These are different numbers
and neither can be derived from the other's stored sums.

**Data.** The read-only Tahoe-100M mirror. Only the five plates that carry a 0.5 µM well
(plate2, 5, 8, 11, 13) were opened — a DMSO well controls its own plate and no other. 412 samples,
50 cell lines, **31,293,704 cells** aggregated out of the mirror's 95,624,334, in one pass of
**1,475 s**. Nothing under `/mnt/S1/sdata/pubdata/Tahoe/` was written.

**What changed.** The same pass also wrote a **fixed single-cell manifest**: a cell is in the
sample if `blake2b(barcode)` falls below a threshold set from its own group's cell count, aiming
at 64 cells per (well, cell line). That rule depends on nothing but the barcode, so the sample is
identical whatever order the shards are read in. **1,266,204 cells** from 20,600 groups, stored as
`log1p(1e4 * count / library)` on the frozen 2,000-gene panel. chemCPA trains on these cells. No
pseudobulk row is ever passed to a model as a cell.

**Result.** The condition table reproduces v1's coverage exactly — 16,521 valid conditions,
9,883 / 3,305 / 3,333 train/val/test, the same two zero-coverage lines, the same four
structure-less drugs — while the truth's RMS against zero moves from v1's 0.1266 to **0.1135**.
`sha256(Y) = ca65f5e0d00cea0416d12e42e18ebf210b92fb386c4779d61ea2e4b339d44c7b`.

chemCPA's three split names map to this study's: `train`→train, `test`→**validation**,
`ood`→**test**. Only `final_eval_cpa.py` opens `ood`, once. `uns["all_DEGs"]` is present and
empty: computing DE genes for a held-out drug would need that drug's held-out responses.

**Meaning.** Every baseline and every model in this study is scored on v2 and nothing is carried
over from v1's scale. Details and the exposure statement are in `DATA_CARD_V2.md`: Tahoe-100M is a
public benchmark that at least one public model has already trained on, so this is a
**pre-registered held-out-drug evaluation inside an already-exposed benchmark**, and what is
protected is the selection procedure, not the secrecy of the data.

---

## E3 — Baselines and the v1 architectures, re-scored on v2

**Purpose.** Give the arms something to beat that costs nothing.

**Result** (validation, 3,305 conditions, 48 lines):

| | macro-ΔRMSE | SNU-423 |
|---|---:|---:|
| no change (predict zero) | 0.10280 | 0.10946 |
| per-line mean of training drugs | 0.09132 | 0.09667 |
| ridge on the same inputs (λ=1, scanned on validation) | **0.09128** | 0.09655 |
| v1 small MLP A, re-trained on v2 | 0.09670 | — |
| v1 small MLP C, re-trained on v2 | 0.09673 | — |

**Meaning.** On this task a per-line constant is already close to the best simple answer, and
ridge adds very little over it (drug-centred r = 0.068). That sets a demanding floor.

---

## E4 — The five pre-run checks (goal §4)

Before any score existed (`reports/chemcpa_moe_v2/checks.json`):

1. **Shared initialisation.** At a seed, P0/P1/P2/P3 hash identically over every module they
   share. Across seeds the hashes differ and the realised hyper-parameters are identical.
   This one was initially vacuous: `ComPert.__init__` takes its `hparams` dict branch only for a
   plain dict, and an OmegaConf node is not one, so `set_hparams_(seed, hparams)` runs and calls
   `torch.manual_seed(model.additional_params.seed)`. Setting the ambient global seed changed
   nothing. The check now varies that effective seed and asserts both directions.
2. **Same function at initialisation.** Every residual's output projection is zero-initialised, so
   P1/P2/P3 produce outputs **bitwise identical** to P0 on real control cells before training.
3. **Parameter parity.** P2 experts `k(dr + r + rd + d)` = 66,688 plus router
   `d_cond·32 + 32 + 32k + k` = 2,212 → 68,900; P1 `2dm + m + d` with m = 134 → 68,998. Relative
   difference **0.14 %**.
4. **The residual receives gradient.** Every residual parameter moves and the loss falls; for P2
   the router and all four experts move and the router does not collapse onto one expert.
5. **Split disjointness.** The treated drug groups of train/validation/test are pairwise disjoint,
   and the validation evaluator draws conditions only from validation drug groups.

---

## E5 — The arms: learning-rate scan, then seeds (goal §4, §5)

**Purpose.** The study's own comparison, on validation only.

**Model.** One base model, three arms, residual at the decoder's last hidden state (width 256):

| arm | residual | parameters |
|---|---|---:|
| P0 | none — the chemCPA prototype | 2,249,533 |
| P1 | size-matched Dense, `d→134→d`, `fc2` zero-init | +68,998 |
| P2 | K = 4 soft-MoE, rank 32, routed on `cell_drug_embedding` (64-d), `w2` zero-init | +68,900 |
| P3 | input-independent fixed experts | **not trained** — see §7 |

Equal opportunity by construction: same cells, same genes, same batches, same official warm-up and
objective, same optimizer with the residual added to chemCPA's own autoencoder parameter group,
same epoch budget. Learning rate was scanned over the official value × {0.3, 1, 3} at seed 42, one
scale selected per arm on validation, then that scale re-run at seeds 43 and 44 — exactly the
protocol the plan fixes.

**Result.** All three arms selected **3× the official learning rate**. Fifteen runs,
`trials.tsv`:

| arm | 0.3× | 1× | 3× | seeds 42 / 43 / 44 at 3× | seed mean ± sd | ensemble |
|---|---:|---:|---:|---|---:|---:|
| P0 | 0.3124 | 0.3371 | **0.2925** | .2925 / .2979 / .3123 | 0.3009 ± 0.0103 | 0.2906 |
| P1 | 0.3244 | 0.3558 | **0.2952** | .2952 / .3222 / .2759 | 0.2977 ± 0.0232 | 0.2844 |
| P2 | 0.3267 | 0.3433 | **0.2891** | .2891 / .2823 / .3030 | 0.2915 ± 0.0105 | 0.2800 |

The shared-initialisation hash is identical across arms within a seed and different across seeds,
as recorded in `trials.tsv`.

**Meaning, stated carefully.** P2's seed mean is 3.12 % better than P0's and 2.10 % better than
P1's, but only **two of three seeds** agree that P2 beats P1, and every arm is about three times
worse than predicting no change. One more fact cuts against P2 and is recorded for that reason:
**P1's best epoch is the last one in four of its five runs**, so P1 was still improving when the
shared budget ended. A margin over a model that had not finished training is not a margin.

---

## E6 — Where the score actually comes from (diagnostic, validation only)

**Purpose.** The primary metric subtracts the **observed** plate DMSO pool from the predicted
treated mean, so its error is exactly `predicted treated profile − observed treated profile`. A
model that cannot reproduce a cell line's absolute expression is charged for that even if its
response were right. Separate the two, on the frozen checkpoints, selecting nothing.

**What changed.** A second control mode: pass the same control cells through the model with the
DMSO index at dose 0 and subtract *that* instead. It cancels any constant per-block reconstruction
offset.

**Result** (`diagnostics.json`, seed means over 42/43/44):

| | P0 | P1 | P2 | reference |
|---|---:|---:|---:|---|
| observed control (the primary) | 0.3009 | 0.2977 | 0.2915 | no change **0.1028** |
| model control (diagnostic) | **0.1044** | **0.1044** | **0.1049** | no change 0.1028 |
| reconstruction's share of the primary | 64–66 % | 62–67 % | 63–66 % | |
| RMS of the predicted response, model control | 0.015–0.028 | 0.017–0.029 | 0.020–0.033 | truth RMS **0.1097** |

chemCPA's own logged objective over the same runs: reconstruction loss 0.168 → 0.110 (P0),
0.168 → 0.117 (P1), 0.163 → 0.124 (P2); drug-adversary loss 0.075 → 0.018; covariate-adversary
loss flat at ≈ 3.90.

**Meaning.** Two separate failures, which the primary metric adds together:

1. About two thirds of each arm's score is **absolute-profile error**. Given a cell line's own
   DMSO cells and asked to reconstruct them, the decoder is off by ≈ 0.30 RMS per gene — nearly
   three times the size of the response it is supposed to predict.
2. Remove that offset and every arm sits **on the no-change baseline** (0.1044–0.1049 against
   0.1028, still slightly worse), with a predicted response only a quarter to a fifth of the true
   magnitude. Under its own control the model is, to the precision this data supports,
   predicting no drug effect.

The reconstruction loss was still falling at the last epoch of every run, so the 10-epoch state
cannot be read as the architecture's limit. §4 of the plan forbids exactly that reading, which is
why E7 exists.

---

## E7 — The same setting, three times the budget, for all three arms equally

**Purpose.** Remove undertraining as an explanation before any conclusion is drawn.

**What changed.** `max_epochs` 10 → 30. Nothing else: same three arms, the learning rate
validation already selected, the same three seeds, the same data, the same objective. No new arm,
metric or search dimension is introduced, and no selection is made on anything but validation.

**Which runs go to the test, declared before the results were seen.** The 30-epoch runs. The
seed and the data order are unchanged, so a 30-epoch run's first ten epochs are the same ten
epochs as before, and its best-on-validation epoch is chosen from a superset that contains the
earlier choice. Using it is therefore the same validation-only selection over a larger range, not
a second selection between two protocols. If the longer budget scores worse on validation, that is
reported as the result; it is not grounds for going back to the shorter one.

**Result.** Filled in by `experiments/chemcpa_moe_v2/budget_cpa.sbatch` (jobs 21421–21423) — see
§8.

---

## X1 — PIE (chemical / Tahoe): not run, and the reason is not the output contract

Full record in `X1_PIE.md`. In short, the two usual blockers do not apply and the real one is
narrower:

```
output_contract:  MATCHABLE   PIE's label is also mean(log1p(CP10k)); it differs from v2's only
                              in using a per-cell-line rather than a per-plate control, which is
                              a computable offset this study already has both terms for
holdout:          VIOLATED    the only public model that has seen chemical data,
                              arcinstitute/PIE_replogle_xdataset, trains on PIE_tahoe100m at
                              dataset_weights.tahoe = 0.68, so it has seen all 70 of this
                              study's held-out test drug groups; no PIE_tahoe* model repo exists
X1 status:        not_run     retraining tahoe_wdataset on this study's split is a second study
```

So X1 is **not_run**, not `output_contract_blocked`. Scoring the public checkpoint would be
scoring a model on its own training data. What running it properly would cost is itemised in
`X1_PIE.md` (`devices: 2`, `max_steps: 5000`, `accumulate_grad_batches: 4`, plus a `pie prep` pass
and the `PIE_sources` knowledge directories). The consequence for this study: the planned external
reference point is missing, so nothing here speaks to how a larger knowledge-conditioned model
would do — the conclusions are about chemCPA-scale models on this data.

---

## §7 — The P3 promotion gate

Pre-registered: P3 is trained only if P2 improves on **both** the prototype and the size-matched
Dense by at least 2 % relative, all three seeds agree in direction, and P2 beats the minimum
baselines. On validation (`selection.json`):

| condition | value | met |
|---|---|---|
| P2 over P0 ≥ 2 % | +3.12 % | yes |
| P2 over P1 ≥ 2 % | +2.10 % | yes |
| three seeds agree P2 beats P1 | 2 of 3 (seed 44 prefers P1) | **no** |
| P2 beats no_change / line_mean / ridge | 0 of 3 | **no** |

`train_P3: false`. P3 was not trained, and the question it was to answer — whether a fixed,
input-independent expert mixture would do as well as a routed one — is reported as
**`routing_effect: not_tested`**, not as a negative result.

One defect in the gate's own code was found and fixed before this ran: the baselines were read as
a flat `{name: number}` map while the file stores one block per baseline, so the filtered dict was
empty and `all({})` made the baseline condition pass vacuously. It now flattens the blocks and
exits non-zero if it finds none. The gate above is the corrected one.

---

## §8 — The extended-budget result

*(to be completed from jobs 21421–21423)*

## §9 — The single test pass

*(to be completed; it opens `ood` once, after every hash is committed)*

---

## Costs actually measured

| stage | compute |
|---|---|
| v2 shard pass (31.3 M cells) | 1,475 s, CPU |
| 15 arm runs at 10 epochs | ≈ 25,500 GPU-s (≈ 7.1 GPU-hours), 1 × A30/A40 per run, peak 80–83 MB allocated |
| control-mode diagnostic | one GPU job over 9 frozen checkpoints |
| 9 arm runs at 30 epochs | see §8 |

Peak allocated memory is small because the model is small; the wall clock is dominated by the
1.27 M-cell epoch, not by the residual. P2's per-epoch cost is within a few per cent of P0's.
