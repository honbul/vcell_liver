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
| `main_effect` | **negative** | on 70 held-out drug groups, all three arms score 0.289–0.294 macro-ΔRMSE against ridge's 0.0919 and a per-line constant's 0.0923. Conditional experts do not rescue this, and neither does the chemCPA prototype: the question "does P2 beat P0 and a size-matched Dense" is answered inside a regime where none of the three beats a per-line constant. |
| `routing_effect` | **not_tested** | the pre-registered promotion gate failed on validation, so P3 — the input-independent fixed expert mixture — was never trained. The routed-versus-fixed question is untouched, not answered. |
| `hcc_effect` | **negative** | SNU-423 (`CVCL_0366`): P0 0.2946, P1 0.2728, P2 0.2882, against ridge 0.0982 and a per-line constant 0.0987. The ordering among arms also flips here, with P1 best. |

One secondary result is **positive and goes the other way**, and is reported as such in §9:
drug-centred correlation, which removes each cell line's mean response, is 0.196 / 0.204 / 0.213
for P0 / P1 / P2 against 0.097 for ridge and 0.028 for the v1 small MLPs. The chemCPA arms read
drug-specific structure better than anything else here while losing badly on scale.

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
`log1p(1e4 * count / library)` on the frozen 2,000-gene panel. Of those, 9,636 belong to groups
that are not in the condition table and are dropped when the AnnData is built, so chemCPA trains
on **1,256,568** cells (765,503 train / 262,289 validation / 228,776 test). No
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

Nine runs, 30 epochs, ≈ 5,200 s each (`trials.tsv`). The first ten epochs reproduce the earlier
runs exactly where the earlier best lay inside them — P0 seed 42 is still 0.29248 at epoch 2, P1
seed 42 still 0.29518 at epoch 9, P2 seed 42 still 0.28913 at epoch 4 — which is also the
reproducibility check for the whole pipeline.

| arm | seeds 42 / 43 / 44 | seed mean | 10-epoch mean | best epochs |
|---|---|---:|---:|---|
| P0 | .2925 / .2908 / .2902 | 0.2912 | 0.3009 | 2 / 21 / 20 |
| P1 | .2952 / .2954 / .2759 | 0.2888 | 0.2977 | 9 / 27 / 9 |
| P2 | .2891 / .2823 / .2847 | **0.2854** | 0.2915 | 4 / 7 / 13 |

Three times the budget bought about 3 % on the primary metric and left every arm roughly three
times worse than predicting no change. The 30-epoch diagnostic (`diagnostics_e30.json`) is
unchanged from the 10-epoch one: reconstruction is still 63–64 % of the score, the model-control
score is still 0.1049–0.1053 against no-change's 0.1028, and the predicted response is still
0.017–0.033 RMS against a true 0.1097. chemCPA's reconstruction loss went from 0.110–0.124 to
0.106–0.111 over the extra twenty epochs.

**Meaning.** Undertraining is no longer available as the explanation for the three-fold gap. It
remains true that these models were trained for 30 epochs and not 300; what the extension shows
is that the gap does not close at the rate a budget argument would need.

The gate was re-scored on these runs and now fails **all four** conditions, not two: P2 over P0
falls to +1.98 % and over P1 to +1.19 %, both under the threshold. `train_P3: false` either way.

---

## §9 — The single test pass

**Purpose.** Score the frozen arms once on the 70 held-out drug groups.

**Data.** chemCPA's `ood` split, opened by `final_eval_cpa.py` and by nothing else. 3,333 test
conditions, 48 cell lines, 70 drug groups. Every checkpoint's SHA-256 is in
`manifests/chemcpa_moe_v2/freeze.json`, and the code was committed before the job ran — the job
refuses to start otherwise.

**Result.**

| | macro-ΔRMSE (seed mean) | 3-seed ensemble | drug-centred r | top-20 direction | SNU-423 |
|---|---:|---:|---:|---:|---:|
| P0 chemCPA prototype | 0.29418 | 0.28334 | 0.1958 | 0.654 | 0.29457 |
| P1 size-matched Dense | 0.29139 | 0.27973 | 0.2037 | 0.662 | **0.27276** |
| P2 K = 4 soft-MoE | **0.28863** | **0.27835** | **0.2131** | 0.670 | 0.28819 |
| v1 small MLP A | 0.09679 | — | 0.0275 | 0.840 | 0.10224 |
| v1 small MLP C | 0.09679 | — | 0.0278 | 0.841 | 0.10220 |
| no change | 0.10427 | — | undefined | 0.0 | 0.11197 |
| per-line mean (training drugs) | 0.09230 | — | 0.0007 | 0.865 | 0.09870 |
| ridge | **0.09193** | — | 0.0968 | 0.866 | **0.09819** |

Paired bootstrap over drug groups, 2,000 resamples, on the 3-seed ensemble predictions:

| | difference | 95 % interval | relative |
|---|---:|---|---:|
| P0 → P1 | 0.00361 | [0.00259, 0.00451] | 1.27 % |
| P0 → P2 | 0.00499 | [0.00401, 0.00591] | 1.76 % |
| P1 → P2 | 0.00139 | [0.00076, 0.00202] | 0.50 % |

Plate diagnostic, 239 blocks: removing each (cell line, plate) block's own mean from both sides
leaves `plate_centred_r` at 0.004 / −0.028 / 0.007 for P0 / P1 / P2 and the plate-centred
macro-ΔRMSE at 0.0818 / 0.0808 / 0.0810 — that is, once a per-block offset is removed the arms
land near the trivial baselines, and essentially none of the remaining agreement is drug-specific
structure within a plate.

**Meaning.**

1. **`main_effect` is negative.** The ordering P2 < P1 < P0 is consistent and its bootstrap
   intervals exclude zero, but the whole ordering sits about three times above every trivial
   baseline. An improvement of 1.8 % inside that regime is not evidence that conditional experts
   help the task; it is evidence that they slightly reduce a dominant error that is not the one
   the study is about. Validation had already refused to promote P2, and the test pass does not
   overturn that.
2. **`hcc_effect` is negative**, and on SNU-423 the arm ordering flips — P1 is best, P2 third.
   With 70 drugs in one cell line, that is the sampling noise a 0.5 % margin lives inside.
3. **The one positive secondary.** Drug-centred correlation is 2–7× higher for the chemCPA arms
   (0.196–0.213) than for ridge (0.097) or the v1 MLPs (0.028), and it rises monotonically
   P0 → P1 → P2. Removing each cell line's mean response, the arms do read drug-specific
   structure, and the conditional experts read slightly more of it than a Dense residual, which
   reads slightly more than the prototype. What they cannot do is place that structure at the
   right absolute level: top-20 direction accuracy is 0.65–0.67 against the baselines' 0.86,
   because a baseline that predicts the per-line mean gets the common directions right for free.

So the honest summary of the study's own question is: **P2 is reproducibly, slightly better than
P1 and P0 on a metric all three fail**, and the mechanism the plan proposed — conditional
residual experts at the shared decoder hidden state — is not shown to help the task. The reason
all three fail is given in §6 and is not about the residual at all: chemCPA, trained natively on
this single-cell sample for up to 30 epochs, reconstructs a cell line's absolute profile to about
0.30 RMS per gene and predicts a drug response about a fifth of the true magnitude.

**What this does not establish.** That chemCPA cannot do this task. The model was trained on
1.26 M cells from 5 plates at one dose for 30 epochs, with the authors' hyper-parameters chosen on
sci-Plex, and its own reconstruction loss was still improving. A study that wanted to answer
"can chemCPA do this" would start from that, not from here.

---

## Costs actually measured

| stage | compute |
|---|---|
| v2 shard pass (31.3 M cells, 3,388 shards) | 1,475 s, CPU |
| 15 arm runs at 10 epochs | 25,550 GPU-s ≈ **7.1 GPU-hours** |
| 9 arm runs at 30 epochs | 46,820 GPU-s ≈ **13.0 GPU-hours** |
| two control-mode diagnostics | 2 GPU jobs over 9 frozen checkpoints each |
| two test passes | ≈ 6 min each (the second only to repair the SNU-423 lookup) |
| total model training | **≈ 20.1 GPU-hours** on one A30 or A40 at a time |

Peak allocated memory is 80.1 MB (P0), 81.6 MB (P1), 82.9 MB (P2); the model is small and the
wall clock is dominated by the 1.26 M-cell epoch, not by the residual. Per-run cost at 30 epochs
is 5,209 s (P0), 5,172 s (P1), 5,225 s (P2) — within 1 % of each other, so the conditional
experts cost essentially nothing in time and 0.3 % in memory over the prototype.

---

## Deviations from the plan, and defects found in this study's own code

Recorded because the plan asks for actually-run / not-run / blocked to be distinguished, and
because every one of these was found after it had already affected something.

| | what happened | what was done |
|---|---|---|
| epoch budget | the plan fixes the learning-rate search but not a budget; 10 epochs left chemCPA's reconstruction loss still falling | raised to 30 for **all three arms equally**, at the already-selected learning rate, with the rule for which runs go to test declared before the results existed (§7) |
| P3 gate code | the baselines were read as a flat `{name: number}` map while the file stores one block per baseline, so the filtered dict was empty and `all({})` passed the baseline condition vacuously | flattened, and the script now exits non-zero if it finds no baseline; the gate in §7 is the corrected one |
| seed check | the first version set the ambient global seed, which `ComPert.set_hparams_` then overrides, so all three "seeds" were the same model | the check now varies the effective seed and asserts both that seeds differ and that hyper-parameters do not |
| evaluator control term | the predicted response first used the mean of the 64 sampled control cells while the truth subtracts the whole plate DMSO pool | `dmso_plate_mean.npy` added, arms restarted |
| final evaluation device | Lightning returns the module to CPU when `fit` ends, while `ComPert.predict` moves inputs to the device string it was built with | `lm.to(dev)` before the final evaluation |
| scoring scripts' arguments | neither the diagnostic nor the test script declared `--patience`/`--seed`, which `build()` reads, so both died at argument parsing | declared in both; the diagnostic failed this way once (job 21420 is the rerun) |
| SNU-423 | looked up as `CVCL_0564` while the manifest and the rest of the study use `CVCL_0366`, so the pre-specified secondary was silently absent from the first test pass | identifier corrected, a missing line is now a hard error, and the pass writes its predictions to the server so no further read-out reopens the test split. **The test split was therefore opened twice.** Nothing was selected, refitted or changed between the two passes; the first pass is committed unmodified so the repair is visible as a diff |
| early stopping | chemCPA's `patience` is its own counter and is not wired into the Lightning loop, so every run goes to `max_epochs` | `budget_limited` redefined to mean "the best epoch is at the end of the budget", and reported per run |
| X1 | PIE was investigated and **not run**; the blocker is the holdout, not the output contract | `X1_PIE.md` |

An earlier statement of mine that runs would "early-stop around epoch 4–5" was wrong for the
reason in the last row, and is corrected here rather than quietly dropped.
