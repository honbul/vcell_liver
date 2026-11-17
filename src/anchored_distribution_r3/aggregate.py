"""Aggregate per-well scores: wells -> condition -> drug, equal weights, drug-block bootstrap.

Seeds are averaged at the well level before aggregation (they are not replicates of the
experimental unit); their spread is reported separately as `seed_sd`. Paired contrasts against the
reference methods are bootstrapped on per-drug differences.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import drug_block_bootstrap
from .setting import run_dir

METRICS = ["ed", "ed_shape", "ed_pca50", "mean_rmse", "var_ratio", "var_logabs", "zero_frac_gen", "zero_frac_obs",
           "zero_frac_mae", "cov_relfro", "cluster_tv", "neg_frac"]


def load(setting, tags):
    fr = [pd.read_csv(run_dir("eval", setting) / f"{t}.tsv.gz", sep="\t").assign(tag=t) for t in tags]
    return pd.concat(fr, ignore_index=True)


def drug_level(df, col):
    w = df.groupby(["line", "drug", "gid"], as_index=False)[col].mean()         # seeds averaged per well
    c = w.groupby(["line", "drug"], as_index=False)[col].mean()                  # well-first
    return c.groupby("drug")[col].mean()                                         # lines equal inside a drug


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="dev")
    ap.add_argument("--tags", required=True)
    ap.add_argument("--out", required=True, help="output prefix under reports/.../tables")
    ap.add_argument("--refs", default="shifted:anchored")
    ap.add_argument("--lines", default="", help="restrict to these line indices (comma-separated)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--cond-table", default="")
    a = ap.parse_args()
    R = load(a.setting, a.tags.split(","))
    if a.lines:
        R = R[R.line.isin([int(x) for x in a.lines.split(",")])]
    R["mv"] = R.method + ":" + R.variant
    cols = [c for c in METRICS if c in R.columns]
    rows = []
    dl = {mv: {c: drug_level(g, c) for c in ["ed", "ed_shape", "ed_pca50"]} for mv, g in R.groupby("mv")}
    for mv, g in R.groupby("mv"):
        row = dict(method=mv, n_wells=int(g.gid.nunique()), n_conditions=int(g.groupby(["line", "drug"]).ngroups),
                   n_drugs=int(g.drug.nunique()), n_lines=int(g.line.nunique()),
                   n_seed_runs=int(g.groupby(["train_seed", "sample_seed"]).ngroups))
        for c in ["ed", "ed_shape", "ed_pca50"]:
            b = drug_block_bootstrap(dl[mv][c].to_frame(), B=a.boot)
            row[c], row[c + "_lo"], row[c + "_hi"] = b["est"], b["lo"], b["hi"]
        per_seed = [float(drug_level(s, "ed").mean()) for _, s in g.groupby(["train_seed", "sample_seed"])]
        row["ed_seed_sd"] = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else np.nan
        row["ed_seed_min"], row["ed_seed_max"] = float(np.min(per_seed)), float(np.max(per_seed))
        for c in cols:
            if c not in row:
                row[c] = float(drug_level(g, c).mean())
        for ref in a.refs.split(","):
            if ref in dl and ref != mv:
                for c in ["ed", "ed_shape", "ed_pca50"]:
                    diff = (dl[mv][c] - dl[ref][c]).dropna()
                    b = drug_block_bootstrap(diff.to_frame(), B=a.boot)
                    k = f"d_{c}_vs_{ref}"
                    row[k], row[k + "_lo"], row[k + "_hi"] = b["est"], b["lo"], b["hi"]
                    if c == "ed":
                        row[f"frac_drugs_better_vs_{ref}"] = float((diff < 0).mean())
        rows.append(row)
    T = pd.DataFrame(rows).sort_values("ed")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    T.to_csv(str(out) + ".tsv", sep="\t", index=False, float_format="%.6g")
    show = ["method", "n_drugs", "n_seed_runs", "ed", "ed_lo", "ed_hi", "ed_shape", "ed_pca50", "mean_rmse", "var_ratio",
            "zero_frac_mae", "cluster_tv"] + [c for c in T.columns if c.startswith("d_ed_vs")]
    print(T[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if a.cond_table:
        C = R.groupby(["mv", "line", "drug"], as_index=False)[cols].mean().rename(columns={"mv": "method"})
        C.to_csv(a.cond_table, sep="\t", index=False, float_format="%.5g")


if __name__ == "__main__":
    main()
