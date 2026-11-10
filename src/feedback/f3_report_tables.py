#!/usr/bin/env python
"""F3: condense LOCO outputs into report-ready headline tables (reads only small TSVs +
assets/f3/loco/per_target_metrics.tsv.gz). Writes reports/feedback/F3/headline_*.tsv."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="reports/feedback/F3")
    ap.add_argument("--assets", default="assets/f3")
    a = ap.parse_args()
    rep = Path(a.rep)
    pb = pd.read_csv(rep / "paired_bootstrap.tsv", sep="\t")
    keep = [
        ("A_unseen_ctx_seen_target", "calibrated_same_target", "no_change"),
        ("A_unseen_ctx_seen_target", "same_target_mean", "no_change"),
        ("A_unseen_ctx_seen_target", "calibrated_same_target", "same_target_mean"),
        ("A_unseen_ctx_seen_target", "ridge_pca", "same_target_mean"),
        ("A_unseen_ctx_seen_target", "nearest_line_same_target", "same_target_mean"),
        ("A_unseen_ctx_seen_target", "global_mean", "no_change"),
        ("Bs_double_ood_simulated", "ridge_pca", "no_change"),
        ("Bs_double_ood_simulated", "ridge_pca", "global_mean"),
        ("Bs_double_ood_simulated", "global_mean", "no_change"),
        ("C_inline_unseen_target", "ridge_pca", "no_change"),
        ("C_inline_unseen_target", "ridge_pca", "global_mean"),
    ]
    rows = []
    for c, p, r in keep:
        x = pb[(pb.condition == c) & (pb.predictor == p) & (pb.reference == r)]
        for m in ["rmse", "pearson_pertcentered", "deg_auprc", "retrieval_rank"]:
            for _, y in x[x.metric == m].iterrows():
                rows.append(dict(condition=c, predictor=p, reference=r, metric=m, heldout=y.heldout,
                                 diff=y.mean_diff, lo=y.ci95_lo, hi=y.ci95_hi, n=y.n_targets))
    h = pd.DataFrame(rows)
    h.to_csv(rep / "headline_paired.tsv", sep="\t", index=False, float_format="%.5f")
    # cross-line vs in-line on identical targets (S_L): A(calibrated, best simple) vs C(ridge) vs Bs(best)
    df = pd.read_csv(Path(a.assets) / "loco" / "per_target_metrics.tsv.gz", sep="\t")
    sub = df[df.in_test_split | (df.condition != "A_unseen_ctx_seen_target")]
    piv = []
    for L, x in sub.groupby("heldout"):
        def get(cond, pred, m):
            v = x[(x.condition == cond) & (x.predictor == pred)].set_index("target")[m]
            return v
        for m in ["rmse", "pearson", "pearson_pertcentered", "deg_auprc"]:
            a_ = get("A_unseen_ctx_seen_target", "calibrated_same_target", m)
            c_ = get("C_inline_unseen_target", "ridge_pca", m)
            b_ = get("Bs_double_ood_simulated", "ridge_pca", m)
            n_ = get("C_inline_unseen_target", "no_change", m)
            common = a_.index.intersection(c_.index)
            piv.append(dict(heldout=L, metric=m, n=len(common),
                            A_calibrated_same_target=a_[common].mean(), C_inline_ridge=c_[common].mean(),
                            Bs_ridge=b_[common].mean(), no_change=n_[common].mean(),
                            A_minus_C=(a_[common] - c_[common]).mean()))
    pd.DataFrame(piv).to_csv(rep / "headline_same_targets_A_vs_C.tsv", sep="\t", index=False, float_format="%.5f")
    # strata condensed: calibrated_same_target minus no_change RMSE and pert-centred pearson by stratum
    st = pd.read_csv(rep / "loco_strata.tsv", sep="\t")
    out = []
    for (L, s, lev), x in st.groupby(["heldout", "stratifier", "level"]):
        x = x.set_index("predictor")
        if "calibrated_same_target" not in x.index:
            continue
        out.append(dict(heldout=L, stratifier=s, level=lev, n=int(x.loc["no_change", "n_targets"]),
                        rmse_no_change=x.loc["no_change", "rmse"], rmse_calibrated=x.loc["calibrated_same_target", "rmse"],
                        rmse_same_target=x.loc["same_target_mean", "rmse"],
                        pertc_same_target=x.loc["same_target_mean", "pearson_pertcentered"],
                        auprc_same_target=x.loc["same_target_mean", "deg_auprc"], auprc_global=x.loc["global_mean", "deg_auprc"]))
    pd.DataFrame(out).to_csv(rep / "headline_strata.tsv", sep="\t", index=False, float_format="%.5f")
    print(h[h.metric.isin(["rmse", "pearson_pertcentered"])].to_string())
    print(pd.DataFrame(piv).to_string())
    print(pd.DataFrame(out).to_string())


if __name__ == "__main__":
    main()
