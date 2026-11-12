# Model comparison — what differs between A, B and C, and what the validation path found

Companion to `RESULTS.md`. This document covers the construction of the comparison and everything
decided on validation. The single test pass is in `RESULTS.md`.

## 1. Only the residual response transform changes

Every model is the same network apart from `R(h)`:

```
basal (2,000)  → cell encoder → z (128)
morgan (2,048) → drug encoder → d (128)
[z, d] (256)   → trunk        → h (256)
                 decoder(h + R(h)) → Δ (2,000)
```

| | `R(h)` | input-dependent? |
|---|---|---|
| **A** Dense | `W₂ GELU(W₁h)`, one bottleneck of width 146 | no extra mechanism; h already carries cell and drug |
| **C** Conditional experts | `Σₖ πₖ(z,d) Aₖ(h)`, 4 experts of rank 32, softmax router on `[z,d]` | yes |
| **B** Fixed experts | `Σₖ softmax(a)ₖ Aₖ(h)`, the same 4 experts, one global `a` | no |

Encoders, trunk, decoder, dropout (0.1 on the common encoders and trunk only), loss, optimiser,
schedule, batching, early stopping and seeds are byte-identical across models. All four experts
are always computed: this is soft routing, not sparse top-k, and no speed claim is made or implied.
No pre-trained weights are loaded anywhere — A, B and C train from scratch.

A's bottleneck width was **not** tuned. `solve_dense_hidden` builds C, counts its parameters, then
scans integer widths and keeps the one closest in total count. That is a pure arithmetic step
executed before training; no score was consulted and no unused dummy parameters were added.

## 2. Parameter parity

| | total | shared | response transform `R` | router |
|---|---:|---:|---:|---:|
| A (hidden 146) | 1,273,186 | 1,198,032 | 75,154 | — |
| C (K=4, r=32) | 1,273,076 | 1,198,032 | 75,044 | 8,356 |
| B (K=4, r=32) | 1,264,724 | 1,198,032 | 66,692 | 4 |

A vs C: **0.0086 %** apart in total and **0.147 %** in the response transform, inside the 1 % and 5 %
the plan allows. B removes 8,352 router parameters from C and keeps the experts.

Parameter parity is not FLOP parity. C computes four expert branches plus a router where A computes
one branch, so C is the slower model at equal parameter count; the measured times are in §6.

A caveat that matters for reading any B result: a fixed convex combination of parallel MLPs can be
rewritten as a single wider Dense MLP. B is therefore the control that removes input dependence
from C — not a third independent model family, and not a separate finding if it wins.

## 3. Pre-run checks

All four required checks pass (`checks.json`), on the rebuilt table:

| check | result |
|---|---|
| no drug name / group / treated well in two splits | 0, 0, 0; 905 other-dose wells of these drugs kept out of training |
| **no identical fingerprint across splits** | 0 in validation, 0 in test (added after the v0 defect) |
| inputs aligned with truth, panel train-only | 2,000 genes, 15 technical excluded, sampled lookups match |
| 64-condition batch is fitted, losses and gradients finite | A 0.0821 → 0.0031, C 0.0803 → 0.0030, no parameter left unmoved |
| C's router and all four experts updated, gate not collapsed | all moved; entropy 1.327 of a possible 1.386 |

## 4. The grid, seed 42 — identical six settings per model

Validation macro ΔRMSE. Lower is better; `no-change` is 0.12618, per-line mean 0.11045, ridge 0.11026.

| lr | wd | A | C | best epoch |
|---|---|---:|---:|---:|
| 3e-4 | 1e-4 | 0.11859 | 0.11845 | 2 |
| 3e-4 | 1e-2 | 0.11858 | 0.11845 | 2 |
| 1e-3 | 1e-4 | 0.11962 | 0.11942 | 1 |
| 1e-3 | 1e-2 | 0.11962 | 0.11942 | 1 |
| **3e-3** | **1e-4** | 0.11632 | 0.11647 | 0 |
| **3e-3** | **1e-2** | **0.11632** | **0.11646** | 0 |

Selected for both models: **lr 3e-3, weight decay 1e-2**, on overall validation macro ΔRMSE at seed
42 and nothing else. The two weight-decay settings differ in the fifth decimal, so weight decay is
doing essentially nothing at this scale.

**Every setting of both models peaks at epoch 0–2 and degrades from there.** That is the dominant
fact of this study and it is identical for the two architectures:

| epoch | A train loss | A val | C train loss | C val |
|---|---:|---:|---:|---:|
| 0 | 0.0240 | 0.11632 | 0.0234 | 0.11646 |
| 1 | 0.0146 | 0.11875 | 0.0145 | 0.11816 |
| 3 | 0.0108 | 0.12030 | 0.0107 | 0.12102 |
| 10 | 0.0074 | 0.12354 | 0.0073 | 0.12248 |
| 30 | 0.0060 | 0.12336 | 0.0058 | 0.12290 |

Training RMSE falls from 0.155 to about 0.077 while validation rises past 0.123, which is worse than
predicting no change at all. With 9,901 training conditions and a 2,000-dimensional output, both
networks memorise the training conditions within a few epochs, and early stopping then hands back a
model that has barely left its initialisation.

Nothing was done about this. The learning rate, schedule, capacity, dropout and patience are the
pre-registered ones, and the plan forbids widening the search in response to a result.

## 5. Seed confirmation and the promotion gate

Selected setting at seeds 42, 43, 44:

| seed | A | C | C over A |
|---|---:|---:|---:|
| 42 | 0.11632 | 0.11646 | −0.127 % |
| 43 | 0.11646 | 0.11646 | +0.001 % |
| 44 | 0.11636 | 0.11644 | −0.065 % |
| **mean** | **0.11638** | **0.11646** | **−0.064 %** |

| gate condition | required | observed | met |
|---|---|---|---|
| mean relative improvement of C over A | ≥ 2 % | −0.064 % | **no** |
| all three seeds in the same direction | 3/3 | 1 of 3 favours C | **no** |
| beats the per-line mean (0.11045) | yes | 0.11646 | **no** |
| beats ridge (0.11026) | yes | 0.11646 | **no** |
| SNU-423 not worse by more than 2 % | ≥ −2 % | −0.153 % | yes |

Four of five fail, so **B was not trained**, exactly as the plan specifies for a failed gate. 16 of
the 24 permitted runs were used; the 8 reserved for B were not spent.

The 2 % threshold was fixed before any score and was not lowered afterwards. The gap between A and
C is smaller than the seed-to-seed spread of either model, so the honest reading is that the two
architectures are indistinguishable here, not that A is better.

## 6. Cost

Measured on one A30, same batch, after training (`resources.tsv`):

| | parameters | active parameters | train time, selected run | inference, batch 256 |
|---|---:|---:|---:|---:|
| A | 1,273,186 | 1,273,186 | 6.4 s | see `resources.tsv` |
| C | 1,273,076 | 1,273,076 (all four experts computed) | 17.4 s | see `resources.tsv` |

C's extra cost buys nothing measurable here. Note that these times cover 31 epochs ending at a model
selected from epoch 0, so they say more about how quickly the recipe stops than about the cost of
converging either architecture.

Peak GPU memory is recorded per run in `resources.tsv` via `torch.cuda.max_memory_allocated`;
`sacct` remains unreachable for this account, so no scheduler-side memory figure is available —
the same limitation the earlier studies recorded.

## 7. What the router actually did

`routing_usage.tsv` holds, for every C run, the mean mixture weight per expert, the entropy, and the
standard deviation of each weight **across inputs**. The last column is the one that matters: a
router can hold a high entropy and still emit nearly the same mixture for every condition, which
would make C behave like B without B ever being trained. These numbers are reported as observed.
No load-balancing or diversity loss was added, and no regularisation search was started in response
to the gate's behaviour.

## 8. What this comparison cannot settle

- **The routing question is untested.** `routing_effect: not_tested`. B was skipped because the gate
  failed, so nothing here says whether input-dependent combination would help a model that worked.
- The comparison is valid — same data, same budget, same code, same seeds — but it is a comparison
  of two models that both lose to a linear fit on the same inputs. A structural difference measured
  between two models in that state does not transfer to a regime where either model works.
- Four experts at rank 32 is one point in a space the plan deliberately did not sweep. "Conditional
  experts do not help here" is a statement about this configuration, this data size and this
  training recipe.
