# Data card — v2: the same conditions, a different within-well aggregation, and real single cells

Two artifacts are built from one new pass over the Tahoe-100M shards:

| | what it is | used by |
|---|---|---|
| `assets/v2/data` | the (cell line, drug) condition table under the **v2** truth | the baselines, the v1 small MLPs, and the scoring of every chemCPA arm |
| `assets/v2/tahoe_cells.h5ad` | 1,266,204 real single cells on the frozen 2,000-gene panel | chemCPA itself, which trains on cells and not on pseudobulk rows |

## 1. Why the shards had to be read again

v1, and every artifact under `prior_assets/r3`, aggregates a well as the log of a pooled mean:

```
v1   E = log1p( 1e4 * sum_cells(count) / sum_cells(library) )
v2   E = mean_cells( log1p( 1e4 * count / library ) )
```

The stored arrays hold `sum_cells(count)` and `sum_cells(library)`. Nothing in them recovers a
per-cell logarithm, so v2 cannot be derived from v1 — it needs the cells. The truth is otherwise
unchanged: treated minus the **same plate's** DMSO pool of the same cell line, replicate wells
weighted equally, `log(mean)` never swapped for `mean(log)` in the other direction either.

The difference is not cosmetic. Against a zero prediction the v2 truth has RMS **0.1135**, against
v1's **0.1266**: taking the log per cell before averaging shrinks the measured response by about a
tenth. Every score in this study is therefore on a different scale from v1's, which is why the
baselines and the v1 architectures are all re-run here rather than carried over.

## 2. What the pass read, and what it did not

Only the five plates that carry a 0.5 µM well — **plate2, plate5, plate8, plate11, plate13** —
were opened. A DMSO well controls its own plate and no other, so the remaining nine plates'
controls are unusable for this dose and their shards were skipped. Within those plates the pass
read the 402 treated wells at 0.5 µM and the 10 DMSO wells, 412 samples in all, across 50 cell
lines: **31,293,704 cells** aggregated out of the mirror's 95,624,334.

The raw mirror was opened read-only. Nothing was downloaded, and no file under
`/mnt/S1/sdata/pubdata/Tahoe/` was written.

One consequence to state plainly: the per-line basal DMSO profile that goes into the models' input
is now pooled over those five plates, where v1's was pooled over all fourteen. The *truth* is
unaffected, because it only ever used the condition's own plate.

## 3. The condition table reproduces v1's coverage exactly

Same 349 structure groups, same 209/70/70 drug split at seed 20261112, same 2,000-gene panel (train
responses only), same cell line order, same inclusion thresholds (≥ 50 cells in the treated well,
≥ 100 cells in the plate's DMSO pool for that line).

| | v2 | v1 |
|---|---:|---:|
| valid conditions | 16,521 | 16,521 |
| train / val / test | 9,883 / 3,305 / 3,333 | 9,883 / 3,305 / 3,333 |
| cell lines with zero coverage | CVCL_1531, CVCL_1571 | the same two |
| drugs with no usable structure | 4 | the same four |

The coverage is identical because the inclusion rules depend on cell counts, which the new
aggregation does not change. Only the numbers inside the table moved.

`sha256(Y) = ca65f5e0d00cea0416d12e42e18ebf210b92fb386c4779d61ea2e4b339d44c7b`.

## 4. The fixed single-cell sample

chemCPA is a model of cells. Feeding it pseudobulk rows dressed as cells would make the comparison
meaningless, so the same pass wrote a sample of real cells.

A cell is in the sample if `blake2b(barcode)` falls below a threshold set from its own group's
known cell count, aiming at **64 cells per (well, cell line)**. That rule depends on nothing but
the barcode: it is identical whatever order the shards are read in, however the work is split
across workers, and however often it is re-run. 1,266,204 cells were selected from 20,600 groups.

Each cell is stored as `log1p(1e4 * count / library)` on the frozen 2,000-gene panel — the same
quantity the v2 truth averages, so the model's target and the study's truth are the same scale.
`cells_meta.tsv` carries every selected cell's barcode, well, cell line, plate and library size, so
the sample is a manifest and not just an array.

## 5. How the cells become chemCPA's AnnData

chemCPA knows three splits and this study needs three, so the mapping is fixed and stated:

| chemCPA split | this study | contents |
|---|---|---|
| `train` | train | the 209 training drug groups' treated cells, plus 70 % of the control cells |
| `test` | **validation** | the 70 validation drug groups' treated cells, plus 30 % of the control cells |
| `ood` | **test** | the 70 held-out test drug groups' treated cells |

Everything that selects a setting reads `test`, which is this study's validation. `ood` is opened
once, by `final_eval_cpa.py`, after the arms are frozen. The names are chemCPA's; the discipline is
the study's.

Control cells carry no drug response, so dividing them between the two splits leaks nothing about a
held-out drug. The division is by a hash of the barcode.

`make_anndata.py` asserts that the treated drug groups of `train`, `test` and `ood` are pairwise
disjoint, and fails the build otherwise.

**Differentially expressed genes.** `uns["all_DEGs"]` is present with empty gene lists for every
category. chemCPA uses it only to score `R2_mean_de`, and computing DE genes for a held-out drug
requires that drug's held-out responses. The field is well formed, the mask is all zeros, and no
held-out truth is read. This study's metrics come from its own evaluator.

## 6. Exposure

Tahoe-100M is public, this project has analysed it before, and at least one public model
(`arcinstitute/PIE_replogle_xdataset`) was trained on all of it. A new split seed does not make it a
blind cohort. The correct name for this evaluation is a **pre-registered held-out drug evaluation
within an already-exposed public benchmark**: what is protected is this study's own selection
procedure, not the secrecy of the data. See `X1_PIE.md` for what that exposure rules out.

Unresolved and not claimed otherwise: train and test conditions share plates and share the DMSO
pools that define their truth; the dose is fixed at 0.5 µM, so nothing here speaks to dose
response; and the 2,000-gene panel is a panel, not the transcriptome.

## 7. Provenance

| | |
|---|---|
| build | `src/chemcpa_moe_v2/build_v2.py` → `assets/v2/raw/{logsum,ncells,cells_X}.npy`, `{groups,cells_meta}.tsv`, `build_record.json` |
| table | `src/chemcpa_moe_v2/prepare_v2.py` → `assets/v2/data`, `manifests/chemcpa_moe_v2/transform.json` |
| cells | `src/chemcpa_moe_v2/make_anndata.py` → `assets/v2/tahoe_cells.h5ad`, `manifests/chemcpa_moe_v2/anndata.json` |
| drug embedding | `src/chemcpa_moe_v2/make_embedding.py`, which calls chemCPA's own `embeddings/rdkit/embedding_rdkit.py` |
| jobs | `experiments/chemcpa_moe_v2/{build_v2,prepare_v2}.sbatch` |

The arrays stay on G1 under
`/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver_chemcpa_moe_v2`; Git holds the code, the
manifests, the hashes and the commands that regenerate them.
