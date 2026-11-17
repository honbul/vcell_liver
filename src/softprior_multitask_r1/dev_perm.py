#!/usr/bin/env python3
"""Real versus permuted pathway prior on the development (inner-fold) score. No outer-test or LCCL label.

For each named model: selection-honest development score with the real Hallmark sets and with each
whole-axis gene-label permutation; the real score's position among the permutations; and a paired
family-bootstrap interval for real minus the mean over permutations.

Usage: dev_perm.py --real-runs A,B,.. --perm-runs C,D,.. --models m1,m2 --tag NAME --out-dir DIR
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_data as R  # noqa: E402
import r1_select as SEL  # noqa: E402
import metrics as A1M  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-runs", required=True)
    ap.add_argument("--perm-runs", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    cfg = R.load_config()
    rd = lambda s: [os.path.join(R.runs_dir(cfg), r) for r in s.split(",")]
    reg = R.load_registry()
    fams = sorted(reg[(reg.study == "LIMORE") & reg.included].family_id.unique())
    b = cfg["metric"]["bootstrap"]
    W = A1M.bootstrap_weights(len(fams), b["n"], b["seed"])
    names = args.models.split(",")
    specs = {m: SEL.model_spec(cfg, m) for m in names}
    seeds = cfg["hallmark"]["permutation_seeds"]

    def scores(prior):
        runs = rd(args.real_runs) if prior == "real" else rd(args.perm_runs)
        inner = SEL.load_inner(runs, prior)
        assert len(inner) == 15, (prior, len(inner))
        nc = sum(not r["converged"] for it in inner.values() for r in it["records"])
        return {m: (SEL.dev_xsel(inner, specs[m]), SEL.dev_xsel(inner, specs[m], fams, W)) for m in names}, nc

    real, _ = scores("real")
    perm, n_fail = {}, 0
    for s in seeds:
        perm[s], nc = scores(f"perm{s:03d}")
        n_fail += nc
    rows, long = [], []
    for m in names:
        pv = np.array([perm[s][m][0] for s in seeds])
        pb = np.mean([perm[s][m][1] for s in seeds], axis=0)
        d = real[m][1] - pb
        lo, hi = A1M.ci(d, *b["ci"])
        rows.append(dict(model=m, real=real[m][0], perm_mean=pv.mean(), perm_sd=pv.std(ddof=1), perm_min=pv.min(),
                         perm_max=pv.max(), n_perm=len(pv), n_perm_at_or_below_real=int((pv <= real[m][0]).sum()),
                         real_minus_perm_mean=real[m][0] - pv.mean(), ci_low=lo, ci_high=hi))
        long += [dict(model=m, prior=f"perm{s:03d}", dev_xsel_macro_nrmse=v) for s, v in zip(seeds, pv)]
        long.append(dict(model=m, prior="real", dev_xsel_macro_nrmse=real[m][0]))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.out_dir, f"{args.tag}_real_vs_perm.tsv"), sep="\t", index=False, float_format="%.6g")
    pd.DataFrame(long).to_csv(os.path.join(args.out_dir, f"{args.tag}_perm_scores.tsv"), sep="\t", index=False, float_format="%.6g")
    print(out.to_string())
    print("permuted-prior solver records not converged:", n_fail)


if __name__ == "__main__":
    main()
