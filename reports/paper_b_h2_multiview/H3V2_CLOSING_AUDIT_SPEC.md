# H3-v2 Closing Audit Spec (Zero-Retraining Paper B Final Bridge)

- **Created**: 2026-11-25 KST
- **Target Output Workspace (Local)**: `/home/gsun/work/vc_liver/paper_b_h2_multiview/`
- **Target Output Workspace (G1)**: `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/paper_b_h2_posthoc/`
- **G1 Python Runtime**: `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_5track/runtime/bin/python`
- **Reference Git Commits (Strictly Read-Only)**:
  - `research/nextgen-h3-v2`: `c4428216f64ca9f165acd2f6d01c5caece2aa36c`
  - `research/followup-h2-e5-upgrade`: `1287c0baa4baba649665c54f2efafa9e7c026eb7`
  - `research/followup-h2-e1e3`: `9d6636632bef164dc761b81565b0c97d4f77b6d0`
  - `research/nextgen-h2-v2`: `dafa4d82ff5e568ace6a58b4275620b34a4061c0`

---

## 0. Hard Guardrails & Non-Negotiable Constraints

1. **Zero model retraining / zero refits / zero modification to archived worktrees**:
   - Do **NOT** run any of the 719 unprobed `H3-v2` neighbors, do **NOT** retrain any neural/GBDT/kernel model, and do **NOT** modify or push any existing Git worktree (`nextgen_h3_v2`, `followup_h2_*`, `nextgen_h2_*`).
   - Preserve all original archived receipts (`H3-v2` `score_receipt.json`, `verification.json`, `RESULTS.md`, and `E5` receipts) untouched. All new outputs go strictly into `/home/gsun/work/vc_liver/paper_b_h2_multiview/` (and G1 mirror `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/paper_b_h2_posthoc/`).
2. **Do NOT redo already completed Track B (`W1–W4`) or `E1–E4–E5` audits**:
   - Reuse the already-verified Track B (`PAPER_B_POSTHOC_AUDIT_AND_CLAIM_LEDGER.md`, `max_abs_diff = 0.0`) and `E1_E4_E5_PAIRED_AUDIT.md` artifacts cleanly.
3. **Cluster execution etiquette (G1)**:
   - Run CPU scoring jobs via Slurm (`sbatch` on `cpu` / `cpu-c3`, e.g., 4–8 cores, 16–32 GB RAM, 30 min walltime).
4. **Mandatory machine-replay gate**:
   - Before computing any new cross-study contrast, verify that loading frozen `H3-v2` forecasts and `E5`/`H2-v2` forecasts reproduces archived `H3V2_METRICS.tsv` and `E5_METRICS.tsv` within `max_abs_diff <= 1e-9` on identical rows and keys. Abort and diagnose if replay exceeds `1e-9`.

---

## 1. Verified Input Paths & Cryptographic Anchors (G1)

| Artifact | Verified G1 Path | Notes |
| :--- | :--- | :--- |
| `H3-v2` Frozen Final Forecasts (240 fits + refs) | `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_v2/h3_v2/final/` | Subdirs: `drug-0..2`, `cell-0..2`, `CTRPv2-0..2`, `GDSC2-0..2`, `PRISM20Q2-0..2`, `external`, each containing `{attention, film, gbdt, gnn, hybrid, references}` |
| `H3-v2` Splits & Manifest | `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_v2/h3_v2/splits/` | `manifest.json` + per-split train/query definitions |
| `H3-v2` Score Artifacts | `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_v2/h3_v2/score/` | `H3V2_METRICS.tsv`, `H3V2_PAIRED_95CI.tsv`, `score_receipt.json`, `drug_units.parquet`, `cell_units.parquet` |
| `E5` / `H2-v2` / `E4` Frozen Forecasts | `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/followup_20261124/h2_e5_upgrade/` | Roles `3..5` = DRP Drug-Cold (`fold 0..2`, 670,830 rows, 1,186 parents), role `6` = External (`GDSC1`, `LIMORE`, `LICOB`); contains `H2-v1`, `H2-v2`, `C1`, `H2-LOSO-Full`, `P1-Minimax`, `P3B-Shared`, `P3B-InScreen`, `H2-v3-Combined` |
| `E1` Frozen Forecasts (`Cchem-int`, `F01-only`) | `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/followup_20261124/h2_e1e3/` | Reused on exact matching row keys |
| `E3` Strengthened `980-family` Roster | `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/followup_20261124/h2_e1e3/registration/cell_roster.parquet` | SHA256 `44b53f6506360786a279f5b38d4dd12d648fc85063f0a74748b1b27ab24b1d52` (`1,014` `cell_id`s -> `991` `lineage_group` -> `980` `family`) |

---

## 2. Work Packages (4 Concise Zero-Retraining Closing Modules)

### A1 — Direct Paired 95% CI: `H3-v2` vs `H2-v2` vs `E5` vs `E1` (`Cchem-int`) on Identical Frozen Rows
*Addresses external `H3-v2` audit §3.2 & §5.5 (closing the H2-v1 vs H2-v2/E5 baseline gap and adding external LINCS-observed/unobserved & LICOB18 breakdowns).*

1. **Exact row-key & truth parity gate**:
   - Join `H3-v2` frozen forecasts (`drug-0..2` and `external`) with `E5`/`E1` frozen forecasts (`roles 3..5` for DRP Drug-Cold, `role 6` for External) on exact `(row_id / source_id, cell_id, compound_id)` keys.
   - Verify 100% row-count parity (`670,830` rows / `1,186` parents on Drug-Cold; exact external panel rows on `GDSC1`, `LIMORE`, `LICOB`, `LICOB18`) and exact truth-value parity (`max_abs_diff = 0.0`).
   - Run machine-replay verification confirming archived `H3V2_METRICS.tsv` (on `H2-v1`, `C1`, `Champion`, 5 families) and archived `E5_METRICS.tsv` (on `H2-v2`, `P3B-Shared`, `P3B-InScreen`, `H2-v3-Combined`) replay within `max_abs_diff <= 1e-9`.
2. **Direct 2,000-bootstrap paired 95% CIs (exact native evaluator, matching `H3-v2` / `E5` cluster rules)**:
   - **Drug-Cold (`1,186` parents)** stratified across `all (1,186)`, `lincs_obs (724)`, and `lincs_unobs (462)`:
     - Compare `H3-v2 Champion (hybrid)`, all 5 individual families (`gbdt`, `film`, `attention`, `gnn`, `hybrid`), and descriptive `Champion-ensemble` directly against:
       1. `H2-v1` (archived reference),
       2. **`H2-v2`** (same-contract modern structured multi-view kernel — primary apples-to-apples comparison!),
       3. **`E1 Cchem-int`** (chemistry + potency intercept baseline),
       4. **`E5 P3B-Shared`** (shared-component cross-platform kernel without known-screen InScreen term),
       5. **`E5 H2-v3-Combined` / `P3B-InScreen`** (explicitly flagged as using registered screen-specific component — contract difference annotated in every table!).
     - Report **both** 3-seed averaged unit-score paired 95% CIs **and** individual seed (seed 0, 1, 2) point estimates + SD so seed variance is never hidden.
   - **External (3 external datasets comprising 4 evaluation panels: `GDSC1`, `LIMORE`, `LICOB`, `LICOB18`)**, stratified by `all`, `lincs_obs`, and `lincs_unobs`:
     - Direct paired 95% CIs of `H3-v2 Champion` & 5 families vs `H2-v1`, **`H2-v2`**, **`E1 Cchem-int`**, **`E5 P3B-Shared`**, and **`E5 H2-v3-Combined`**.
     - Specifically report per-seed `GDSC1` NDCG@5 spread (`SD = 0.0407`) alongside 3-seed mean!
   - **`68-drug held/core68` panel (`CTRPv2/GDSC2/PRISM20Q2`)**:
     - Add direct paired 95% CI of `H3-v2 Champion` and all 5 families vs **current `H2`** (using the exact `core_score.py` rank-gain NDCG / pooled interaction $\rho$ evaluator, strictly separated into its own table with explicit formula metadata so it is never conflated with native metrics!).
3. **Deliverables**:
   - `tables/H3V2_VS_H2V2_E5_METRICS.tsv`
   - `tables/H3V2_VS_H2V2_E5_PAIRED_95CI.tsv`
   - `tables/H3V2_HELD68_VS_H2_PAIRED_95CI.tsv`

---

### A2 — Strengthened `980-Family` Biological Independence Audit on `H3-v2` Splits
*Addresses external `H3-v2` audit §5.1 (checking whether E3's Cellosaurus-merged 980 families cross `H3-v2`'s 991-lineage train/query splits).*

1. **Metadata-only split cross-tabulation**:
   - Hash-verify `cell_roster.parquet` (`44b53f65...`).
   - For every `H3-v2` split (`cell-0..2`, `CTRPv2-0..2`, `GDSC2-0..2`, `PRISM20Q2-0..2`, `drug-0..2`), map all train and query `cell_id`s to both `lineage_group` (991) and strengthened `family` (980).
   - Report per split: `n_train_cells`, `n_query_cells`, `lineage_overlap` (expected 0 on Cell-Cold/held), **`family_overlap_count`**, exact list of any crossing `family` IDs & `cell_id`s, and exact number & percentage of query rows belonging to crossing families.
2. **Zero-retraining sensitivity check (if `family_overlap_count > 0` on any Cell-Cold or held split)**:
   - Re-score frozen `H3-v2` predictions on `Cell-Cold` and `held68` **excluding** any query rows whose strengthened `family` appeared in the corresponding split's training set, AND evaluate `980-family` cluster bootstrap sensitivity on Drug-Cold/Cell-Cold/GDSC1 NDCG@5.
   - Report exact before/after metric delta and verify whether any conclusion changes (and state clearly that excluding query rows is a post-hoc leak-free sensitivity check, not a replacement for split-level independence).
3. **Deliverables**:
   - `tables/H3V2_FAMILY980_SPLIT_OVERLAP.tsv`
   - `tables/H3V2_FAMILY980_SENSITIVITY.tsv`

---

### A3 — Standalone Verification of `parts=True` Decomposition Discrepancy & Native Evaluator Null Surfaces
*Addresses external `H3-v2` audit §4.1 & §5.6 without modifying `nextgen_h3_v2`.*

1. **`parts=True` inventory & counterexample receipt**:
   - Programmatically scan `nextgen_h3_v2/reports/nextgen_h3_v2/final_lock.json` to list every `(split, family)` whose locked config used `parts=True` (verify exact count = 4: `GDSC2-1` FiLM/GNN, `PRISM20Q2-0` FiLM/GNN; verify **0/3 Drug-Cold splits** and **0/1 External Champion** used `parts=True`).
   - Run the deterministic unbalanced toy counterexample against read-only `h3v2.objectives.component_targets` and log exact `max|y - (baseline + c + residual)|` in the audit receipt.
2. **Native evaluator null-surface audit**:
   - Evaluate read-only `nextgen_h2.metrics` on exact synthetic null surfaces (`constant`, `drug_only`, `cell_only`, pure additive `c + d`) on both dense and realistic sparse masks, plus inspect why archived `drug_mean` had tiny nonzero `interaction_spearman` (`-0.0019` on Drug-Cold, `-0.0505` on GDSC1, `+0.0349` on LICOB — distinguish floating-point zero-tolerance leakage vs donor/screen aggregation!).
   - Verify whether applying a strict zero-tolerance clamp (`1e-9`) to native interaction residuals changes any `H3-v2` or `E5` model contrast by more than `1e-6`.
3. **Deliverables**:
   - `tables/H3V2_PARTS_AND_NULL_AUDIT.tsv`
   - `tables/h3v2_closing_audit_receipt.json`

---

### A4 — Paper B Pre-Drafting Unified Bridge & Interpretation Guardrails
Create `/home/gsun/work/vc_liver/paper_b_h2_multiview/H3V2_CLOSING_AUDIT_AND_PAPER_B_BRIDGE.md` uniting A1–A3 with Track B (`PAPER_B_POSTHOC_AUDIT_AND_CLAIM_LEDGER.md`):
1. **5-Line Executive Verdict Header** (Question / What changed / Key paired CIs / Verdict / Paper B readiness).
2. **Exact Apples-to-Apples Ranking Table** (`H2-v1` vs `H2-v2` vs `E1 Cchem-int` vs `E5 P3B-Shared` vs `E5 H2-v3-Combined` vs `H3-v2 Champion/5 Families` across Drug-Cold `all`/`LINCS724`/`Unobs462` and External `GDSC1`/`LIMORE`/`LICOB`/`LICOB18`, with contract differences explicitly annotated).
3. **Mandatory Interpretation Corrections for Paper B**:
   - Preserve archived `score_receipt.json` intact while phrasing the scientific conclusion accurately as **`NO-GO within the frozen boundary-1 search budget`** (never claim universal architectural impossibility or absolute ceiling proof; explicitly note 719 unprobed neighbors, 72/75 winners from boundary-1, and 13/75 near iteration cap).
   - Frame `H3-v2` accurately as a **comparison of five complete modern modeling recipes** (explicitly noting differing loss/assay-weighting and unprotected linear inner selection `0.45/0.30/0.25` behind Cell-Cold FiLM NDCG drop, plus the 4 peripheral `parts=True` splits).
   - Separate native evaluator formulas (`min–max` NDCG gain, per-drug interaction $\rho$) from `68-drug held/core68` formulas (`rank-gain` NDCG, pooled interaction $\rho$) cleanly.
   - Report exact `980-family` split overlap and null-surface audit outcomes transparently.

---

## 3. Verification Checklist (Definition of Done)

- [ ] Machine replay of archived `H3V2_METRICS.tsv` and `E5_METRICS.tsv` passes at `max_abs_diff <= 1e-9`.
- [ ] All SHA256 hashes (inputs + new tables + receipt) recorded in `tables/h3v2_closing_audit_receipt.json`.
- [ ] Zero modifications to any existing Git worktree (`nextgen_h3_v2`, `followup_h2_*`, `nextgen_h2_*`).
- [ ] All outputs mirrored cleanly between G1 (`/mnt/S1/sdata/workbench/Virtual_Cell/gssong/paper_b_h2_posthoc/`) and local (`/home/gsun/work/vc_liver/paper_b_h2_multiview/`).
