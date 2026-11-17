#!/usr/bin/env python3
"""Development diagnostics from inner-fold losses only: solution rank per configuration and per-drug
selection-honest scores of named models (which drugs gain or lose).

Usage: diag_dev.py --runs DIR[,DIR..] --models m1,m2,.. --pairs a:b,.. --tag NAME --out-dir DIR [--prior real]
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--pairs", default="")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prior", default="real")
    args = ap.parse_args()
    cfg = R.load_config()
    data = R.load_data(cfg)
    inner = SEL.load_inner([os.path.join(R.runs_dir(cfg), r) for r in args.runs.split(",")], args.prior)
    assert len(inner) == 15

    rec = pd.DataFrame([r for it in inner.values() for r in it["records"]])
    if len(rec):
        rk = rec.groupby("config")["rank"].agg(["min", "median", "max"]).reset_index()
        rk["iterations_max"] = rec.groupby("config").iterations.max().values
        rk.to_csv(os.path.join(args.out_dir, f"{args.tag}_rank_by_config.tsv"), sep="\t", index=False)
        p = rec.config.str.split("|", expand=True)
        rec["tau_fraction"] = p[3].astype(float)
        print(rec.groupby("tau_fraction")["rank"].describe().to_string())

    names = args.models.split(",")
    per = {m: SEL.dev_xsel_per_drug(inner, SEL.model_spec(cfg, m)).mean(axis=0) for m in names}
    tab = pd.DataFrame(per, index=pd.Index(data.drugs, name="drug_id"))
    tab.to_csv(os.path.join(args.out_dir, f"{args.tag}_per_drug_dev.tsv"), sep="\t", float_format="%.6g")
    rows = []
    for pair in filter(None, args.pairs.split(",")):
        a, b = pair.split(":")
        d = tab[a] - tab[b]
        rows.append(dict(model=a, reference=b, mean_difference=d.mean(), n_drugs_better=int((d < 0).sum()),
                         n_drugs_worse=int((d > 0).sum()), worst_drug=d.idxmax(), worst=d.max(),
                         best_drug=d.idxmin(), best=d.min(), q10=d.quantile(.1), q90=d.quantile(.9),
                         n_worse_by_0p01=int((d > 0.01).sum()), n_better_by_0p01=int((d < -0.01).sum())))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.out_dir, f"{args.tag}_per_drug_pairs.tsv"), sep="\t", index=False, float_format="%.6g")
    print(out.to_string())
    if "train_mean" in tab:
        skill = 1 - tab.div(tab["train_mean"], axis=0)
        print(skill.describe().T.to_string())
        print(skill.corr(method="spearman").round(2).to_string())
        for pair in filter(None, args.pairs.split(",")):
            a, b = pair.split(":")
            print(pair, "spearman of (a-b) with skill of b:",
                  round(float((tab[a] - tab[b]).corr(skill[b], method="spearman")), 3))


if __name__ == "__main__":
    main()
