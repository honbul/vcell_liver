# Data card — Tahoe-100M, 0.5 µM, held-out drugs in seen cell lines

Built by `src/moe_drug_v1/prepare.py` from the already-aggregated R3 assets. The 100M-cell raw
mirror was **not** re-read and nothing was re-downloaded. Manifests: `manifests/moe_drug_v1/`.

## 1. What one row is

```
input  x = pooled DMSO RNA profile of cell line l  (2,000 genes, z-scored)
         + Morgan fingerprint of drug d            (radius 2, 2,048 bits)
truth  y = mean RNA change of line l under drug d at 0.5 µM   (2,000 genes)

one training row = one (cell line, drug) condition
```

Individual cells are never expanded into rows. 50 lines × 349 drug groups is 17,450 possible
conditions; **16,521 (94.7 %)** pass QC. All of them fall inside the 48 retained lines, which is
98.6 % of that 48 × 349 block — the two dropped lines contribute nothing at all.

## 2. Truth, exactly as computed

Reused unchanged from `src/feedback/r3_prep.py`, so the definition is identical to the earlier
study and no new transform was introduced:

```
E[l,w,g] = log1p( mean over cells in (line l, well w) of CP10k counts )
Δ[l,w,g] = E[l,w,g] − E[l, DMSO pool of plate(w), g]
y[l,d,g] = equal-weight mean of Δ over that drug group's valid wells in line l
```

`log(mean)` is never swapped for `mean(log)`. Replicate wells of one drug get equal weight
regardless of how many cells they hold, and no loss weighting by exposure or cell count is applied
anywhere downstream.

## 3. Inclusion rules, fixed before any response was read

| Rule | Value | Effect |
|---|---|---|
| Dose | exactly 0.5 µM | 402 of 1,316 treated wells |
| Drug structure | valid canonical SMILES required | 4 drugs dropped, never zero-filled |
| Treated well | ≥ 50 cells of that line | per-condition mask |
| Plate DMSO pool | ≥ 100 cells of that line on the same plate | per-condition mask |
| Replicates | equal weight | 349 groups from 398 wells |
| Cell line | ≥ 100 / 30 / 30 valid drugs in train / val / test | 48 of 50 lines kept |
| SNU-423 | ≥ 40 valid drugs in val and in test | met with 69 and 70 |

The four drugs without a usable structure are `Erdafitinib`, `Sacubitril/Valsartan`,
`Selinexor` and `Verteporfin` (the first and third carry a trailing space in the source table and
no SMILES; the second is a two-component product).

**Two cell lines were dropped and it was not close.** NCI-H2122 and NCI-H596 have median treated-well
counts of 21 and 13 cells and DMSO totals of 610 and 345 cells, so they fail both cell-count rules on
every single well — 0 of 398. They are rare in this pool, not excluded for their responses.

Three further lines are thinned but kept: NCI-H661 (347/398 wells pass), SW 1088 (330/398) and
SW 1271. Every kept line clears the 100/30/30 minimum.

## 4. Drug grouping, and the defect that was found and fixed

Groups are **stereochemistry-free canonical parent SMILES**: largest organic fragment, uncharged,
then stereocentres removed. 349 groups from 398 wells; 25 groups carry more than one drug name.

The grouping unit has to be at least as coarse as what the model's input can distinguish. The first
build grouped by stereo-aware SMILES and split Canagliflozin — recorded twice in Tahoe, once with
stereocentres annotated and once without — across train and validation. A radius-2 Morgan
fingerprint is not chirality-aware, so the two records produce a **byte-identical input vector**.
That is leakage, not a close analogue. Grouping by the stereo-free key merges them; `checks.py` now
fails outright if any evaluation drug's fingerprint matches a training drug's at Tanimoto ≥ 0.999,
and after the fix there are **none** in either validation or test. The defect, its scope and the v0
numbers are kept in `superseded_v0/`.

PubChem CID is used only as a cross-check, never to split a structure that is already identical.
24 groups carry more than one CID — salt, hydrate and registration variants of one structure — and
**no CID appears in two different groups**, so the structural key and the registry never
contradict each other in the direction that would cause leakage.

## 5. Genes

11,132 candidates survive the earlier study's measurability filter (mean CP10k ≥ 0.1 in the 48
non-liver DMSO pools). A fixed technical list then removes 15 of them — `MT-` prefixed,
rRNA families (`RNA5S`, `RNA18S`, `RNA28S`, `RNA45S`, `RNA5-8S`), `RN7SK`, `RN7SL`, `RMRP`, `RPPH1`.
Uncertain biotypes were left in rather than guessed at.

The **2,000-gene panel is the top variance of train-split responses only** (9,883 train conditions).
No validation or test response touched the selection. The same 2,000 genes are both the basal input
and the prediction target for every model, baselines included. This is a 2,000-gene panel, not the
transcriptome.

## 6. Split

Drug groups, 60/20/20, seed 20261112, stratified by the modal plate of each group's wells so no
split inherits a skewed plate mix:

| plate | groups | train | val | test |
|---|---:|---:|---:|---:|
| plate2 | 84 | 50 | 17 | 17 |
| plate5 | 79 | 47 | 16 | 16 |
| plate8 | 67 | 40 | 13 | 14 |
| plate11 | 91 | 55 | 18 | 18 |
| plate13 | 28 | 17 | 6 | 5 |
| **total** | **349** | **209** | **70** | **70** |

A test drug is held out **everywhere**: all 48 cell lines, all replicate wells, and its 0.05 µM and
5 µM rows too. `checks.py` confirms 0 drug names, 0 drug groups and 0 treated wells in two splits,
and that 905 other-dose wells of these drugs stay out of training.

SNU-423, the pre-specified HCC line, has 207 / 69 / 70 valid drugs — its train responses are
deliberately *in* the training set, so this is neither an HCC zero-shot nor a leave-one-cell-line-out
design.

## 7. This is a drug-identity holdout, not a scaffold holdout

Structurally similar analogues may well sit in train, and that is by design. Nearest-train Tanimoto
of the evaluation drugs:

| | n | mean | median | p90 | max | ≥ 0.5 | ≥ 0.7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| val | 70 | 0.323 | 0.267 | 0.604 | 0.883 | 15.7 % | 4.3 % |
| test | 70 | 0.315 | 0.254 | 0.598 | 0.873 | 12.9 % | 5.7 % |

Reported as description only. No drug was removed on the basis of these numbers, before or after
seeing any score.

## 8. How large the signal is

The truth itself has a mean per-condition magnitude (RMSE against zero) of **0.1266**, median 0.110,
10th–90th percentile 0.078–0.200. So "predict no change" already sits at ~0.126, and the whole
contest is over the part above that.

## 9. Exposure, and what stays confounded

Tahoe is a public dataset this project has analysed before. A new split seed does **not** make it a
blind cohort. The correct name for this evaluation is a *pre-registered held-out drug evaluation
within Tahoe*: what is protected is this study's selection procedure, not the secrecy of the data.

Unresolved, and not claimed otherwise:

- Train and test conditions share plates and share the DMSO pools that define their truth. Holding
  out drugs does not remove a per-plate measurement environment.
- The earlier study measured DMSO-to-DMSO correlation of 0.76 between untreated wells, so a
  substantial shared component exists in this truth regardless of drug.
- 239 `(cell line, plate)` blocks on each of validation and test carry ≥ 5 evaluated drugs, which is
  what the fixed plate diagnostic in `evaluate.py` reads out. That diagnostic bounds how much
  agreement survives removing a per-block offset; it does not prove the plate effect is gone.
- Dose is fixed at 0.5 µM, so nothing here speaks to dose response.

## 10. Provenance

`manifests/moe_drug_v1/sources.json` records the SHA-256 of every input file, the R3 build metadata,
and the RDKit / NumPy / pandas / Python versions. Large arrays (`Y.npy`, `basal.npy`, `fp.npy`,
checkpoints, predictions) stay on the server under
`/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver_conditional_moe_v1`; Git holds the code, the
manifests, the small result tables and the commands that regenerate everything.
