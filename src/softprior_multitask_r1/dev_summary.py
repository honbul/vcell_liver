#!/usr/bin/env python3
"""Development summary from inner-fold losses only (no outer-test label, no LCCL label is read).

For every model named in the config (`models`) that can be formed from the configurations present in
the given run directories: development score, family-bootstrap interval, chosen configurations, and
paired differences for the listed pairs. Also writes the score of every single fixed configuration.

Usage: dev_summary.py --runs DIR[,DIR2..] --models m1,m2,... [--pairs a:b,c:d] --tag NAME --out-dir DIR
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_data as R  # noqa: E402
import r1_models as M  # noqa: E402
import r1_select as SEL  # noqa: E402
import metrics as A1M  # noqa: E402


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
    runs = [os.path.join(R.runs_dir(cfg), r) for r in args.runs.split(",")]
    inner = SEL.load_inner(runs, args.prior)
    assert len(inner) == 15, f"expected 15 internal training sets, found {len(inner)}"
    reg = R.load_registry()
    fams = sorted(reg[(reg.study == "LIMORE") & reg.included].family_id.unique())
    b = cfg["metric"]["bootstrap"]
    W = A1M.bootstrap_weights(len(fams), b["n"], b["seed"])
    has_fam = all("fam_loss" in it for it in inner.values())
    os.makedirs(args.out_dir, exist_ok=True)

    rows, boots, chosen_rows = [], {}, []
    for name in args.models.split(","):
        spec = SEL.model_spec(cfg, name)
        score, per_set, chosen = SEL.dev_score(inner, spec)
        row = dict(model=name, select=spec["select"], n_candidates=len(SEL.candidates(spec, next(iter(inner.values()))["keys"])),
                   dev_macro_nrmse=score, set_sd=float(np.std(list(per_set.values()), ddof=1)))
        if has_fam:
            boots[name] = SEL.dev_boot(inner, spec, fams, W)
            row["ci_low"], row["ci_high"] = A1M.ci(boots[name], *b["ci"])
        rows.append(row)
        cnt = collections.Counter(M.key_str(k) for ks in chosen.values() for k in ks)
        for k, n in sorted(cnt.items(), key=lambda kv: -kv[1]):
            chosen_rows.append(dict(model=name, config=k, n_chosen=n, fraction=n / sum(cnt.values())))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(args.out_dir, f"{args.tag}_dev_summary.tsv"), sep="\t", index=False, float_format="%.6g")
    pd.DataFrame(chosen_rows).to_csv(os.path.join(args.out_dir, f"{args.tag}_dev_chosen.tsv"), sep="\t", index=False,
                                     float_format="%.4g")
    print(summ.to_string())

    if args.pairs and has_fam:
        d_rows = []
        s = summ.set_index("model").dev_macro_nrmse
        for pair in args.pairs.split(","):
            a, bb = pair.split(":")
            d = boots[a] - boots[bb]
            lo, hi = A1M.ci(d, *b["ci"])
            d_rows.append(dict(model=a, reference=bb, dev_difference=s[a] - s[bb], ci_low=lo, ci_high=hi,
                               boot_fraction_below_zero=float((d < 0).mean())))
        dd = pd.DataFrame(d_rows)
        dd.to_csv(os.path.join(args.out_dir, f"{args.tag}_dev_differences.tsv"), sep="\t", index=False, float_format="%.6g")
        print(dd.to_string())

    # every single configuration, fixed across training sets and drugs (no selection)
    keys = next(iter(inner.values()))["keys"]
    prof = []
    for i, k in enumerate(keys):
        vals = [float(np.sqrt(it["loss"][it["keys"].index(k)]).mean()) for it in inner.values()]
        prof.append(dict(config=M.key_str(k), dev_macro_nrmse=float(np.mean(vals)), set_sd=float(np.std(vals, ddof=1))))
    pd.DataFrame(prof).to_csv(os.path.join(args.out_dir, f"{args.tag}_dev_fixed_configs.tsv"), sep="\t", index=False,
                              float_format="%.6g")
    # solver record
    rec = [r for it in inner.values() for r in it["records"]]
    meta = dict(n_solver_records=len(rec), n_not_converged=int(sum(not r["converged"] for r in rec)),
                max_iterations=int(max([r["iterations"] for r in rec], default=0)),
                seconds_total=float(sum(m["seconds"] for it in inner.values() for m in it["meta"])))
    with open(os.path.join(args.out_dir, f"{args.tag}_dev_solver.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(meta)


if __name__ == "__main__":
    main()
