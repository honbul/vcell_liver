"""R10 v2: exact replay check. The v2 evaluation re-scores v1 heads/baselines on the same triplets;
rows that exist in a v1 run must reproduce the v1 numbers (evaluation code path unchanged except
for added key/diagnostic columns)."""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd

K = ["line", "drug", "episode", "rep_endpoint", "rep_query", "direction", "method", "seed"]
M = ["comp_tv", "ed", "mmd", "within_mean_rmse", "within_ed", "dmean_rmse_gene", "head_comp_tv",
     "assembly_tv_head_vs_reread"]


def read(pattern):
    fs = sorted(glob.glob(pattern))
    return pd.concat([pd.read_csv(f, sep="\t") for f in fs], ignore_index=True) if fs else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="name=v1_glob::v2_glob::train_contract(or 'any')::methods(comma)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for p in args.pairs:
        name, rest = p.split("=", 1)
        g1, g2, tc, meth = rest.split("::")
        a, b = read(g1), read(g2)
        meth = meth.split(",")
        a = a[a.method.isin(meth) & (a.is_oracle == 0)]
        b = b[b.method.isin(meth) & (b.is_oracle == 0)]
        if tc != "any":
            b = b[b.train_contract == tc]
        m = a.merge(b, on=K, suffixes=("_v1", "_v2"))
        r = dict(check=name, n_v1=len(a), n_v2=len(b), n_matched=len(m))
        for c in M:
            if f"{c}_v1" in m:
                d = (m[f"{c}_v1"] - m[f"{c}_v2"]).abs()
                both_nan = m[f"{c}_v1"].isna() & m[f"{c}_v2"].isna()
                r[f"maxabs_{c}"] = float(d.max()) if d.notna().any() else np.nan
                r[f"nan_mismatch_{c}"] = int((m[f"{c}_v1"].isna() ^ m[f"{c}_v2"].isna()).sum())
                _ = both_nan
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(args.out, sep="\t", index=False, float_format="%.3g")
    print(out.to_string())


if __name__ == "__main__":
    main()
