# P0 vs P1 vs P2 — the comparison itself

`RESULTS.md` is the study. This file is the narrower thing the plan asks for: what exactly was
held equal between the three arms, what differed, and what the difference bought.

## 1. What is identical

One base model is built, and the residual is attached to it. Everything outside the residual is
the same object:

| | held equal | how it is checked |
|---|---|---|
| architecture | public chemCPA at `43e830eb`, unmodified — encoder, dose-amortized drug embedding, covariate embedding, decoder, `[mean | softplus(var)]` head, both adversaries | nothing is deleted or renamed; the prototype is P0 |
| hyper-parameters | the authors' `config/model/hparams/default.yaml`, copied verbatim | `configs/chemcpa_moe_v2/hparams_official.json`; the realised hparams are compared across seeds in check 1 |
| objective | GaussianNLL, minus `reg_adversary` × drug adversary and `reg_adversary_cov` × covariate adversary, adversaries updated every third global step with a gradient penalty | unchanged; the residual is added to chemCPA's own autoencoder parameter group |
| warm-up / schedule | the official `StepLR(step_size_lr=6, gamma=0.9)` and the official doser | unchanged |
| initial weights | every shared module starts from the same values at a given seed | SHA-256 over the state dict with `residual.*` excluded; identical across arms, different across seeds (`trials.tsv`, `base_init_sha256`) |
| initial function | P1 and P2 compute **bitwise** what P0 computes before training | every residual's output projection is zero-initialised; check 2 runs real control cells through all arms and compares exactly |
| data | the same 1,266,204 cells, the same 2,000 genes, the same batches, the same split | one AnnData, one `PerturbationDataModule` |
| opportunity | the same learning-rate ladder (official × 0.3 / 1 / 3), the same seeds (42/43/44), the same epoch budget (10, then 30 for all three) | `trials.tsv` |

## 2. What differs

| arm | what is inserted at the decoder's last hidden state (width 256) | parameters |
|---|---|---:|
| **P0** | nothing | 2,249,533 |
| **P1** | `h + W2 σ(W1 h)`, hidden 134, `W2` zero-init | +68,998 |
| **P2** | `h + Σ_k π_k(c) · (h W1_k) W2_k`, K = 4 experts of rank 32, router softmax over `c = cell_drug_embedding` (64-d), `W2` zero-init | +68,900 (66,688 experts + 2,212 router) |
| **P3** | the same experts with `π` a learned constant, independent of the input | **not trained** |

Parameter parity is 0.14 % relative. P1 exists precisely so that "more parameters in that place"
is not the explanation for anything P2 does.

The single structural difference between P1 and P2 is therefore **whether the residual's
transformation depends on which cell line and which drug it is acting for**. That is the
hypothesis.

## 3. What the difference bought

Validation, selected setting (official lr × 3), 30 epochs, seed mean of per-seed scores:

| | P0 | P1 | P2 |
|---|---:|---:|---:|
| macro-ΔRMSE | 0.29116 | 0.28882 | **0.28539** |
| seeds where this arm beats P1 | — | — | 2 of 3 |
| 3-seed ensemble | 0.28005 | 0.27678 | 0.27476 |

Test, one pass:

| | P0 | P1 | P2 |
|---|---:|---:|---:|
| macro-ΔRMSE | 0.29418 | 0.29139 | **0.28863** |
| drug-centred r | 0.1958 | 0.2037 | **0.2131** |
| top-20 direction | 0.654 | 0.662 | 0.670 |
| SNU-423 | 0.29457 | **0.27276** | 0.28819 |
| vs P2, paired bootstrap over drug groups | +1.76 % [0.00401, 0.00591] | +0.50 % [0.00076, 0.00202] | — |
| training time, 30 epochs | 5,209 s | 5,172 s | 5,225 s |
| peak allocated | 80.1 MB | 81.6 MB | 82.9 MB |

So: **P2 is consistently a little better than P1, and P1 a little better than P0**, in the same
direction on validation and test, on the primary metric and on drug-centred correlation, at
essentially no extra cost in time or memory. The margin over P1 is 0.5–1.2 %.

## 4. Why that is not a positive result

Three reasons, in order of how much they matter.

**The regime.** Ridge on the same inputs scores 0.09193 and a per-line constant 0.09230. All
three arms are near 0.29. A 0.5 % improvement in a number that is three times worse than a
constant is an improvement in the dominant error, which §6 of `RESULTS.md` shows is absolute
profile reconstruction, not response prediction. The hypothesis was about the response.

**The pre-registered gate said no.** On validation P2 clears neither 2 % threshold at the final
budget (+1.98 % over P0, +1.19 % over P1), only two of three seeds agree that it beats P1, and it
beats none of the three baselines. All four conditions fail. The margin reported above was not
promoted, and P3 was not trained.

**P1 was still improving when the shared budget ended** in four of its five 10-epoch runs, and its
best 30-epoch seed (0.27589 at seed 44) is the best single validation number any arm produced. A
margin over a model that had not finished training is not a margin, and that seed is exactly the
one where the P2-beats-P1 ordering flips — on validation and again on SNU-423.

## 5. What would, and would not, settle it

Settles it: a setting where chemCPA itself is competitive with a per-line constant. Then the
difference between a routed and an unrouted residual would be measured against a model that is
doing the task, and the 0.5 % would mean something. Nothing in this study's data or budget
reached that setting.

Does not settle it: more seeds at this budget, more learning rates, a bigger K, a different
insertion point, or adding P3 anyway. The plan forbids all of them for the same reason — the
failure is not a shortage of variants — and this study did not run them.

## 6. Where the numbers are

`trials.tsv` (30-epoch runs) · `trials_e10.tsv` (the 10-epoch stage) · `selection.json` and
`selection_e10.json` (the per-arm selection and the gate) · `diagnostics.json` and
`diagnostics_e30.json` (the control-mode split) · `test_results.json` (the single test pass) ·
`mlp_test/` (the v1 architectures and the baselines on test) ·
`manifests/chemcpa_moe_v2/freeze.json` (every scored checkpoint's SHA-256).
