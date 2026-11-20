# Superseded first build (v0) — kept as the record of a data-contract defect

These files are the output of the first data build and its full validation path. They are **not**
the study's results. They are preserved because the plan requires found implementation errors to be
recorded rather than quietly overwritten (plan s9 step 3, agent rule 7).

## What was wrong

v0 grouped drugs by the canonical parent SMILES **with stereochemistry**. Tahoe records Canagliflozin
twice, once with its stereocentres annotated (`Canagliflozin (hemihydrate)`) and once without
(`Canagliflozin`), so the two records produced two different group keys and the split seed sent one
to train and the other to validation.

The model's drug input is a radius-2 Morgan fingerprint, which is not chirality-aware. The two
records therefore produce a **byte-identical input vector**. That is an identical training input
sitting in the evaluation split — leakage, not a close analogue.

```
val   : Canagliflozin               Cc1ccc(C2OC(CO)C(O)C(O)C2O)cc1Cc1ccc(-c2ccc(F)cc2)s1
train : Canagliflozin (hemihydrate) Cc1ccc([C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)cc1Cc1...
Tanimoto(val, train) = 1.000
```

## Scope of the damage

- One drug group of 71 in validation. The test split contained **no** identical-fingerprint pair
  (its highest nearest-train Tanimoto was 0.873), and the test split was never opened under v0.
- Validation is the split that selects settings, so every v0 selection number is affected in
  principle, even though one drug in seventy-one cannot move a macro average much.

## The fix

Grouping now uses the **stereochemistry-free** canonical parent SMILES, so the grouping unit is at
least as coarse as what the input representation can distinguish. `checks.py` gained a hard check
that no evaluation drug's fingerprint is identical (Tanimoto >= 0.999) to any training drug's; the
checks fail outright if one is.

Everything downstream — the table, the panel, the split, the baselines, all 16 runs — was rebuilt
from scratch after the fix. The v0 numbers below are listed only so the two can be compared.

## v0 numbers, for the record

| | validation macro delta-RMSE |
|---|---|
| no-change | 0.12534 |
| per-line mean | 0.10917 |
| ridge | 0.10911 |
| A Dense (lr 3e-3, wd 1e-2) | 0.11490 |
| C Conditional experts (lr 3e-3, wd 1e-2) | 0.11501 |

v0 gate outcome: mean relative C-over-A improvement +0.018% against a 2% threshold, seeds not in
agreement (42 favoured A, 43 and 44 favoured C), and both models behind the per-line mean and ridge.
B was therefore not going to be trained under v0 either.
