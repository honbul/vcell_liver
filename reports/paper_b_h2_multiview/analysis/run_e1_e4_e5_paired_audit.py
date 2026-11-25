"""Zero-retraining direct paired 2,000-bootstrap audit: E1 Cchem-int & F01-only vs E4 vs E5.

Strictly read-only on archived frozen forecasts from E1, E4, and E5.
Zero refits, zero hyperparameter changes, zero changes to any existing worktree.
Uses the exact same `followup_h2.inference` (DRAWS=2000, seed=495123) engine
used in archived E1, E4, and E5 official scorers, and verifies exact machine
replay against archived E1_METRICS.tsv, E1_PAIRED_95CI.tsv, E5_METRICS.tsv,
and E5_PAIRED_95CI.tsv.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from followup_h2.common import panel_hash, sha256
from followup_h2.e1 import arms as e1_arms
from followup_h2.inference import DRAWS, measure, summarize
from followup_h2_e4.data import OUTCOMES, V1_ASSETS, cpu_threads, registry, train_hash
from followup_h2_e4.forecast import licob18
from followup_h2_e5.forecast import P1_OFFSET, P1_PERSONAL, P2_OFFSET
from nextgen_h2.assets import KEY, parquet_lock
from nextgen_h2.run import META, TRAIN, read_json
from nextgen_h2.score import adult_licob, external_labels
from vc_build.external import native_sensitivity


E1_ROOT = Path("/mnt/S1/sdata/workbench/Virtual_Cell/gssong/followup_20261124/h2_e1e3")
E4_ROOT = Path("/mnt/S1/sdata/workbench/Virtual_Cell/gssong/followup_20261124/h2_e4_loso")
E5_ROOT = Path("/mnt/S1/sdata/workbench/Virtual_Cell/gssong/followup_20261124/h2_e5_upgrade")

COMMON_ARMS = (
    "C1",
    "H2-v1",
    "H2-v2",
    "F01-only",
    "Cchem-int",
    "H2-LOSO-Full",
    "P1-Minimax",
    "P1-RelMinimax",
    "P2-Decoupled",
    "P3A-ScreenDebiased",
    "P3B-Shared",
    "P3C-EqualScreenWeight",
    "H2-v3-Combined",
)
DRP_EXTRA_ARMS = ("P3B-InScreen", "C2", "C3")

COMPONENTS = {
    "Cchem-int": "rc",
    "F01-only": "zero",
    "P1-Minimax": P1_PERSONAL,
    "P2-Decoupled": P1_PERSONAL,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_contrasts(arms: tuple[str, ...]) -> list[tuple[str, dict[str, float]]]:
    pairs: list[tuple[str, str]] = []
    # Primary group A: every arm vs E1 Cchem-int
    for a in arms:
        if a != "Cchem-int":
            pairs.append((a, "Cchem-int"))
    # Primary group B: key models vs E1 F01-only (potency intercept only, zero personalization)
    for a in (
        "H2-v2",
        "H2-LOSO-Full",
        "H2-v3-Combined",
        "P3B-Shared",
        "P3C-EqualScreenWeight",
        "P1-Minimax",
        "P2-Decoupled",
        "P3A-ScreenDebiased",
        "Cchem-int",
    ):
        if a in arms and a != "F01-only":
            pairs.append((a, "F01-only"))
    if "P3B-InScreen" in arms:
        pairs.append(("P3B-InScreen", "F01-only"))
    # Primary group C: direct E5 vs E4 and E5 vs H2-v2 cross-checks
    for b in ("H2-LOSO-Full", "H2-v2", "C1"):
        for a in (
            "H2-v3-Combined",
            "P3B-Shared",
            "P3C-EqualScreenWeight",
            "P1-Minimax",
            "P2-Decoupled",
            "P3A-ScreenDebiased",
        ):
            if a in arms and b in arms and a != b:
                pairs.append((a, b))
    if "P3B-InScreen" in arms:
        for b in ("H2-LOSO-Full", "H2-v2", "P3B-Shared", "C1", "C2", "C3"):
            pairs.append(("P3B-InScreen", b))
        for b in ("C2", "C3"):
            pairs.append(("H2-v3-Combined", b))
    pairs.append(("H2-LOSO-Full", "H2-v2"))
    pairs.append(("H2-LOSO-Full", "C1"))
    pairs.append(("H2-v2", "H2-v1"))
    pairs.append(("H2-v2", "C1"))

    # Deduplicate preserving order
    seen = set()
    out = []
    for a, b in pairs:
        name = f"{a} - {b}"
        if name not in seen:
            seen.add(name)
            out.append((name, {a: 1.0, b: -1.0}))
    return out


def load_verified_role(e1_part: int, e4_idx: int, e5_idx: int) -> tuple[pd.DataFrame, dict]:
    e1_dir = E1_ROOT / "e1-forecast" / str(e1_part)
    e1_rec = read_json(e1_dir / "receipt.json")
    e1_df = parquet_lock(e1_dir / "forecast.parquet", e1_rec["forecast_sha256"])

    e4_dir = E4_ROOT / "forecasts" / str(e4_idx)
    e4_rec = read_json(e4_dir / "receipt.json")
    e4_df = parquet_lock(e4_dir / "predictions.parquet", e4_rec["predictions_sha256"])

    e5_dir = E5_ROOT / "forecasts" / str(e5_idx)
    e5_rec = read_json(e5_dir / "receipt.json")
    e5_df = parquet_lock(e5_dir / "predictions.parquet", e5_rec["predictions_sha256"])

    # Strict identity assertions across all three archived experiments
    assert e1_df[META].equals(e5_df[META]), f"META mismatch E1({e1_part}) vs E5({e5_idx})"
    assert e4_df[META].equals(e5_df[META]), f"META mismatch E4({e4_idx}) vs E5({e5_idx})"
    assert np.array_equal(e1_df["lincs_observed"].to_numpy(), e5_df["lincs_observed"].to_numpy())
    assert np.array_equal(e4_df["lincs_observed"].to_numpy(), e5_df["lincs_observed"].to_numpy())

    f11_diff = float(np.max(np.abs(e1_df["F11"].to_numpy(float) - e5_df["H2-v2"].to_numpy(float))))
    f00_diff = float(np.max(np.abs(e1_df["F00"].to_numpy(float) - e5_df["H2-v1"].to_numpy(float))))
    e4_diff = float(np.max(np.abs(e4_df["H2-LOSO-Full"].to_numpy(float) - e5_df["H2-LOSO-Full"].to_numpy(float))))
    assert f11_diff < 1e-6, f"F11 vs H2-v2 diff too large: {f11_diff}"
    assert f00_diff < 1e-6, f"F00 vs H2-v1 diff too large: {f00_diff}"
    assert e4_diff < 1e-6, f"E4 vs E5 H2-LOSO-Full diff too large: {e4_diff}"

    # Merge E1-only columns directly onto the E5 dataframe (exact aligned order)
    merged = e5_df.copy()
    for col in ("i0", "i1", "r0", "r1", "ci", "rc", "F01-only", "Cchem-int"):
        merged[col] = e1_df[col].to_numpy()
    merged["zero"] = 0.0

    proof = {
        "e1_part": e1_part,
        "e4_idx": e4_idx,
        "e5_idx": e5_idx,
        "rows": int(len(merged)),
        "roster_sha256": panel_hash(merged),
        "e1_forecast_sha256": e1_rec["forecast_sha256"],
        "e4_predictions_sha256": e4_rec["predictions_sha256"],
        "e5_predictions_sha256": e5_rec["predictions_sha256"],
        "max_abs_diff_E1_F11_vs_E5_H2v2": f11_diff,
        "max_abs_diff_E1_F00_vs_E5_H2v1": f00_diff,
        "max_abs_diff_E4_vs_E5_LOSO": e4_diff,
    }
    return merged, proof


def aggregate_licob_combined(frame: pd.DataFrame, license_table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate donor replicates identically to BOTH E1 and E5 official scorers."""
    # Exact drug-constant offsets in both E1 (i1, ci) and E5 (__p1_offset, __p2_offset)
    const_cols = ["i0", "i1", "ci", P1_OFFSET, P2_OFFSET]
    offsets = frame.groupby(KEY)[const_cols].first()
    for col in const_cols:
        assert frame.groupby(KEY)[col].nunique().max() == 1, f"Offset {col} not drug-constant"

    drop_cols = ["F01-only", "P1-Minimax", "P2-Decoupled"]
    block = adult_licob(frame.drop(columns=drop_cols), license_table)
    for col in const_cols:
        block[col] = block[KEY].map(offsets[col])

    # Reconstruct exact E1 & E5 offset-bearing arms post-averaging (matches both E1 & E5 identically;
    # note Cchem-int is averaged directly by adult_licob just as in followup_h2/score.py L137-143!)
    block["zero"] = 0.0
    block["F01-only"] = block["i1"]
    block["P1-Minimax"] = block[P1_PERSONAL] + block[P1_OFFSET]
    block["P2-Decoupled"] = block[P1_PERSONAL] + block[P2_OFFSET]
    return block


def score_cohort(
    panels: list[tuple[int, pd.DataFrame]],
    groups: pd.Series,
    cohort: str,
    arms: tuple[str, ...],
) -> tuple[list[dict], list[dict], dict]:
    pairs = build_contrasts(arms)
    arm_meta = {a: {"arm": a} for a in arms}
    points_all, ci_all, support = [], [], {}

    for scope in ("all", "lincs_observed", "lincs_unobserved"):
        selected = [
            (f, df if scope == "all" else df.loc[df.lincs_observed.eq(scope == "lincs_observed")])
            for f, df in panels
        ]
        pooled = pd.concat([df for _, df in selected], ignore_index=True)
        if pooled.empty:
            continue
        unique = {
            "rows": int(len(pooled)),
            "parents": int(pooled[KEY].nunique()),
            "cells": int(pooled.cell_id.nunique()),
            "scaffolds": int(pooled[KEY].map(groups).fillna(pooled[KEY]).nunique()),
            "lineages": int(pooled.lineage_group.nunique()),
        }
        support[scope] = unique
        tag = {
            "cohort": cohort,
            "support": scope,
            "exposure": "held_under_DRP_nested" if cohort == "DRP" else "previously_exposed_external_audit",
        }
        blocks = measure(selected, groups, arms, components=COMPONENTS)
        pts, cis = summarize(blocks, tag, arm_meta, pairs)
        for row in pts:
            if row["screen"] == "macro":
                row.update(unique)
        points_all.extend(pts)
        ci_all.extend(row | {"screen": "macro", "fold": "all"} for row in cis)
        # Also keep per-screen-fold intervals for DRP
        if len(blocks) > 1:
            for b in blocks:
                _, local = summarize([b], tag, arm_meta, pairs)
                ci_all.extend(row | {"screen": b.screen, "fold": b.fold} for row in local)
        print(f"[SCORED] {cohort:8s} | {scope:16s} | {unique}", flush=True)
    return points_all, ci_all, support


def drug_potency_diagnostic(ext_cohorts: dict[str, pd.DataFrame], arms: tuple[str, ...]) -> pd.DataFrame:
    """Mechanistic check: correlation between predicted drug mean potency and observed drug mean sensitivity."""
    rows = []
    for cohort in ("GDSC1", "LIMORE", "LICOB", "LICOB18"):
        df = ext_cohorts[cohort]
        for scope in ("all", "lincs_observed", "lincs_unobserved"):
            sub = df if scope == "all" else df.loc[df.lincs_observed.eq(scope == "lincs_observed")]
            if sub.empty:
                continue
            by_drug = sub.groupby(KEY)[["truth", *arms]].mean()
            y = by_drug["truth"].to_numpy(float)
            n_drugs = len(by_drug)
            for arm in arms:
                p = by_drug[arm].to_numpy(float)
                if np.std(p) < 1e-12 or np.std(y) < 1e-12:
                    rho, pval = np.nan, np.nan
                else:
                    res = spearmanr(y, p)
                    rho, pval = float(res.statistic), float(res.pvalue)
                # Also compute mean within-cell SD of predictions vs drug-mean SD (intercept vs personal variance ratio)
                drug_var = float(np.var(p))
                resid = sub[arm].to_numpy(float) - sub.groupby(KEY)[arm].transform("mean").to_numpy(float)
                personal_var = float(np.var(resid))
                rows.append({
                    "cohort": cohort,
                    "support": scope,
                    "n_drugs": n_drugs,
                    "arm": arm,
                    "drug_mean_spearman": rho,
                    "drug_mean_pval": pval,
                    "drug_mean_var": drug_var,
                    "within_drug_personal_var": personal_var,
                    "intercept_var_share": drug_var / max(drug_var + personal_var, 1e-12),
                })
    return pd.DataFrame(rows)


def verify_exact_replay(
    metrics_df: pd.DataFrame,
    ci_df: pd.DataFrame,
    e1_metrics_path: Path,
    e1_ci_path: Path,
    e5_metrics_path: Path,
    e5_ci_path: Path,
) -> dict:
    """Verify exact machine replay against archived E1 and E5 official TSV tables."""
    # Map support naming difference between E1 ("LINCS-observed") and E5 ("lincs_observed")
    e1_scope_map = {"all": "all", "LINCS-observed": "lincs_observed", "LINCS-unobserved": "lincs_unobserved"}
    e1_arm_map = {"F00": "H2-v1", "F11": "H2-v2", "F01-only": "F01-only", "Cchem-int": "Cchem-int"}

    # 1. Check vs archived E5_METRICS.tsv (macro rows)
    e5_m = pd.read_csv(e5_metrics_path, sep="\t")
    e5_m = e5_m[e5_m["screen"].eq("macro") & e5_m["cohort"].isin(["DRP", "GDSC1", "LIMORE", "LICOB", "LICOB18"])]
    cur_m = metrics_df[metrics_df["screen"].eq("macro")].set_index(["cohort", "support", "arm", "metric"])["value"]
    e5_diffs = []
    for _, r in e5_m.iterrows():
        key = (r["cohort"], r["support"], r["arm"], r["metric"])
        if key in cur_m.index:
            e5_diffs.append((abs(float(cur_m.loc[key]) - float(r["value"])), key, float(r["value"]), float(cur_m.loc[key])))
    e5_diffs.sort(reverse=True)

    # 2. Check vs archived E5_PAIRED_95CI.tsv (macro rows, excluding unique_top5_sets which has separate rng call order)
    e5_c = pd.read_csv(e5_ci_path, sep="\t")
    e5_c = e5_c[e5_c["screen"].eq("macro") & e5_c["cohort"].isin(["DRP", "GDSC1", "LIMORE", "LICOB", "LICOB18"])]
    cur_c = ci_df[ci_df["screen"].eq("macro")].set_index(["cohort", "support", "contrast", "metric", "resampling"])
    e5_ci_diffs = []
    for _, r in e5_c.iterrows():
        if r["metric"] == "unique_top5_sets":
            continue
        key = (r["cohort"], r["support"], r["contrast"], r["metric"], r["resampling"])
        if key in cur_c.index:
            row = cur_c.loc[key]
            d = max(abs(float(row[c]) - float(r[c])) for c in ("delta", "lo95", "hi95"))
            e5_ci_diffs.append((d, key))
    e5_ci_diffs.sort(reverse=True)

    # 3. Check vs archived E1_METRICS.tsv (macro rows for Cchem-int, F01-only, F00, F11)
    e1_m = pd.read_csv(e1_metrics_path, sep="\t")
    e1_m = e1_m[e1_m["screen"].eq("macro")]
    e1_diffs = []
    for _, r in e1_m.iterrows():
        arm = e1_arm_map.get(r["arm"])
        if arm is None:
            continue
        sup = e1_scope_map[r["support"]]
        key = (r["cohort"], sup, arm, r["metric"])
        if key in cur_m.index:
            e1_diffs.append((abs(float(cur_m.loc[key]) - float(r["value"])), key, float(r["value"]), float(cur_m.loc[key])))
    e1_diffs.sort(reverse=True)

    # 4. Check vs archived E1_PAIRED_95CI.tsv (F11 - Cchem-int, F11 - F01-only, F01-only - Cchem-int)
    e1_c = pd.read_csv(e1_ci_path, sep="\t")
    e1_contrast_map = {
        "F11 - Cchem-int": "H2-v2 - Cchem-int",
        "F11 - F01-only": "H2-v2 - F01-only",
        "F11 - F00": "H2-v2 - H2-v1",
    }
    e1_ci_diffs = []
    for _, r in e1_c.iterrows():
        if r["metric"] == "unique_top5_sets":
            continue
        cname = e1_contrast_map.get(r["contrast"])
        if cname is None:
            continue
        sup = e1_scope_map[r["support"]]
        key = (r["cohort"], sup, cname, r["metric"], r["resampling"])
        if key in cur_c.index:
            row = cur_c.loc[key]
            d = max(abs(float(row[c]) - float(r[c])) for c in ("delta", "lo95", "hi95"))
            e1_ci_diffs.append((d, key))
    e1_ci_diffs.sort(reverse=True)

    return {
        "e5_metrics_checked": len(e5_diffs),
        "e5_metrics_max_abs_diff": e5_diffs[0][0] if e5_diffs else None,
        "e5_metrics_worst_case": str(e5_diffs[0]) if e5_diffs else None,
        "e5_ci_checked": len(e5_ci_diffs),
        "e5_ci_max_abs_diff": e5_ci_diffs[0][0] if e5_ci_diffs else None,
        "e5_ci_worst_case": str(e5_ci_diffs[0]) if e5_ci_diffs else None,
        "e1_metrics_checked": len(e1_diffs),
        "e1_metrics_max_abs_diff": e1_diffs[0][0] if e1_diffs else None,
        "e1_metrics_worst_case": str(e1_diffs[0]) if e1_diffs else None,
        "e1_ci_checked": len(e1_ci_diffs),
        "e1_ci_max_abs_diff": e1_ci_diffs[0][0] if e1_ci_diffs else None,
        "e1_ci_worst_case": str(e1_ci_diffs[0]) if e1_ci_diffs else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cpu_threads()
    args.out.mkdir(parents=True, exist_ok=True)
    started = utc_now()

    assert sha256(TRAIN) == train_hash(), "TRAIN hash mismatch"
    reg = registry()

    # Load DRP 3 folds (E1 parts 0,1,2 == E4/E5 roles 3,4,5) and External (E1 part 3 == E4/E5 role 6)
    proofs = []
    drp_raw = []
    for fold in range(3):
        df, proof = load_verified_role(e1_part=fold, e4_idx=fold + 3, e5_idx=fold + 3)
        drp_raw.append((fold, df))
        proofs.append(proof)
    ext_df, ext_proof = load_verified_role(e1_part=3, e4_idx=6, e5_idx=6)
    proofs.append(ext_proof)

    # Read native labels once (identical to E5 official scorer)
    with TRAIN.open("rb") as h:
        source = pd.read_parquet(h, columns=[*META, *OUTCOMES]).sort_values("row_id", ignore_index=True)
    values = pd.Series(native_sensitivity(source).to_numpy(), index=source.row_id)
    del source
    drp_panels = [(f, df.assign(truth=df.row_id.map(values))) for f, df in drp_raw]
    for f, df in drp_panels:
        assert np.isfinite(df["truth"]).all(), f"Non-finite truth in DRP fold {f}"

    labels, license_table, ext_groups = external_labels(V1_ASSETS, read_json(V1_ASSETS / "data_receipt.json"))
    label_map = labels.set_index("row_id").truth
    assert set(label_map.index) == set(ext_df.row_id) and np.isfinite(label_map).all()
    ext_labeled = ext_df.assign(truth=ext_df.row_id.map(label_map))

    ext_cohorts: dict[str, pd.DataFrame] = {}
    for name, block in ext_labeled.groupby("source_id", sort=True):
        if name == "LICOB":
            block = aggregate_licob_combined(block, license_table)
            ext_cohorts["LICOB18"] = licob18(block)
        ext_cohorts[name] = block

    all_points, all_cis, all_supports = [], [], {}
    drp_arms = (*COMMON_ARMS, *DRP_EXTRA_ARMS)
    pts, cis, sup = score_cohort(drp_panels, reg.scaffold_block, "DRP", drp_arms)
    all_points.extend(pts)
    all_cis.extend(cis)
    all_supports["DRP"] = sup

    for cohort in ("GDSC1", "LIMORE", "LICOB", "LICOB18"):
        pts, cis, sup = score_cohort([(0, ext_cohorts[cohort])], ext_groups, cohort, COMMON_ARMS)
        all_points.extend(pts)
        all_cis.extend(cis)
        all_supports[cohort] = sup

    metrics_df = pd.DataFrame(all_points)
    ci_df = pd.DataFrame(all_cis)
    potency_df = drug_potency_diagnostic(ext_cohorts, COMMON_ARMS)

    # Note: explicitly flag unique_top5_sets CIs as descriptive-only in metadata; keep raw column in TSV for replay audit
    metrics_path = args.out / "E1_E4_E5_METRICS.tsv"
    ci_path = args.out / "E1_E4_E5_PAIRED_95CI.tsv"
    potency_path = args.out / "E1_E4_E5_DRUG_POTENCY_DIAGNOSTIC.tsv"

    metrics_df.to_csv(metrics_path, sep="\t", index=False, na_rep="NA")
    ci_df.to_csv(ci_path, sep="\t", index=False, na_rep="NA")
    potency_df.to_csv(potency_path, sep="\t", index=False, na_rep="NA")

    # Verify exact machine replay against archived E1 & E5 TSVs on G1
    replay_audit = verify_exact_replay(
        metrics_df=metrics_df,
        ci_df=ci_df,
        e1_metrics_path=E1_ROOT / "e1-score/E1_METRICS.tsv",
        e1_ci_path=E1_ROOT / "e1-score/E1_PAIRED_95CI.tsv",
        e5_metrics_path=E5_ROOT / "score/E5_METRICS.tsv",
        e5_ci_path=E5_ROOT / "score/E5_PAIRED_95CI.tsv",
    )

    receipt = {
        "audit_name": "E1_Cchem_int_vs_E4_vs_E5_zero_retraining_paired_bootstrap",
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "retrained_models": 0,
        "bootstrap_draws": DRAWS,
        "bootstrap_seed": 495123,
        "resampling_units": {
            "spearman": "transitive_scaffold_block (Bayesian cluster bootstrap)",
            "interaction_spearman": "transitive_scaffold_block (Bayesian cluster bootstrap)",
            "ndcg_at_5": "lineage_group (Bayesian cluster bootstrap)",
            "precision_at_5": "lineage_group (Bayesian cluster bootstrap)",
            "unique_top5_sets": "descriptive_only_no_bootstrap_claim",
        },
        "role_parity_proofs": proofs,
        "exact_replay_vs_archived": replay_audit,
        "supports": all_supports,
        "output_sha256": {
            metrics_path.name: sha256(metrics_path),
            ci_path.name: sha256(ci_path),
            potency_path.name: sha256(potency_path),
        },
    }
    receipt_path = args.out / "e1_e4_e5_paired_audit_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print("\n=== EXACT REPLAY AUDIT ===")
    print(json.dumps(replay_audit, indent=2))
    print("\n=== DONE ===", receipt_path, flush=True)


if __name__ == "__main__":
    main()
