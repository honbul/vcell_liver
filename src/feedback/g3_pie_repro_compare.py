#!/usr/bin/env python
"""G3 / G1-a: compare our `pie eval` outputs with the published per-fold CSVs on the HF model card.

For every (device, fold, setting) with our metrics_best_auprc.csv: the 'all' row vs published,
and the per-pair granular rows (same pairs required) -> max |diff|, mean diff, Pearson.
Match rule (stated in the report): |diff| <= 1e-6 on every 'all' metric = exact; <= 0.005 =
numerically close (expected for a different accelerator / bf16 kernel); otherwise mismatch.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FOLDS = ["hepg2", "jurkat", "k562", "rpe1"]
SETTINGS = ["unseen_ctx", "unseen_pert", "unseen_ctx_pert"]
METRICS = ["binary_auprc", "sig_jaccard", "direction_match", "spearman_nsig", "spearman_lfc",
           "discrimination_score_l1"]
PAIR = ["binary_auprc", "sig_jaccard", "direction_match", "spearman_lfc", "discrimination_score_l1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="published dir: <fold>/eval/<setting>/")
    ap.add_argument("--runs", required=True, help="our runs root: <device>/replogle_wdataset/<fold>/eval/")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows, pairs = [], []
    for dev in ("cuda", "cpu"):
        for fold in FOLDS:
            for s in SETTINGS:
                ours = Path(args.runs) / dev / "replogle_wdataset" / fold / "eval" / s
                pub = Path(args.models) / fold / "eval" / s
                if not (ours / "metrics_best_auprc.csv").is_file():
                    continue
                mo = pd.read_csv(ours / "metrics_best_auprc.csv").set_index("context")
                mp = pd.read_csv(pub / "metrics_best_auprc.csv").set_index("context")
                d = {k: float(mo.loc["all", k] - mp.loc["all", k]) for k in METRICS}
                mx = max(abs(v) for v in d.values())
                status = "exact" if mx <= 1e-6 else ("close" if mx <= 0.005 else "mismatch")
                rows.append(dict(device=dev, fold=fold, setting=s, n_rows=int(mo.loc["all", "n_rows"]),
                                 n_rows_pub=int(mp.loc["all", "n_rows"]), status=status, max_abs_diff=mx,
                                 **{f"ours_{k}": float(mo.loc["all", k]) for k in METRICS},
                                 **{f"pub_{k}": float(mp.loc["all", k]) for k in METRICS},
                                 **{f"diff_{k}": v for k, v in d.items()}))
                go = pd.read_csv(ours / "granular_best_auprc.csv")
                gp = pd.read_csv(pub / "granular_best_auprc.csv")
                m = go.merge(gp, on=["context", "perturbation"], suffixes=("_o", "_p"), how="outer",
                             indicator=True)
                for k in PAIR:
                    a, b = m[f"{k}_o"].to_numpy(float), m[f"{k}_p"].to_numpy(float)
                    ok = ~np.isnan(a) & ~np.isnan(b)
                    pairs.append(dict(device=dev, fold=fold, setting=s, metric=k,
                                      n_both=int((m["_merge"] == "both").sum()),
                                      n_only_ours=int((m["_merge"] == "left_only").sum()),
                                      n_only_pub=int((m["_merge"] == "right_only").sum()),
                                      nan_mismatch=int((np.isnan(a) != np.isnan(b)).sum()),
                                      max_abs_diff=float(np.abs(a[ok] - b[ok]).max()) if ok.any() else np.nan,
                                      mean_diff=float((a[ok] - b[ok]).mean()) if ok.any() else np.nan,
                                      frac_exact=float((np.abs(a[ok] - b[ok]) <= 1e-6).mean()) if ok.any() else np.nan,
                                      pearson=float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else np.nan))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    r = pd.DataFrame(rows)
    r.to_csv(out / "repro_vs_published.tsv", sep="\t", index=False, float_format="%.6f")
    pd.DataFrame(pairs).to_csv(out / "repro_vs_published_pairs.tsv", sep="\t", index=False, float_format="%.6f")
    if len(r):
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(r[["device", "fold", "setting", "status", "max_abs_diff", "ours_binary_auprc", "pub_binary_auprc",
                     "ours_direction_match", "pub_direction_match"]].to_string())
    # published 4-fold mean +- sd (ddof=1) next to ours, per setting and device (only if all 4 folds)
    agg = []
    for (dev, s), g in r.groupby(["device", "setting"]) if len(r) else []:
        if len(g) == 4:
            agg.append(dict(device=dev, setting=s,
                            **{f"ours_{k}": f"{g[f'ours_{k}'].mean():.4f} ± {g[f'ours_{k}'].std(ddof=1):.4f}" for k in METRICS},
                            **{f"pub_{k}": f"{g[f'pub_{k}'].mean():.4f} ± {g[f'pub_{k}'].std(ddof=1):.4f}" for k in METRICS}))
    pd.DataFrame(agg).to_csv(out / "repro_vs_published_4fold.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
