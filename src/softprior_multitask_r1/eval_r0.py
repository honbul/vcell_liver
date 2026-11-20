#!/usr/bin/env python3
"""Stage R1-0 scoring: (a) the R1 scorer reproduces Track A's committed numbers from Track A's committed
out-of-fold tables; (b) Track A predictors rerun on the fixed-ID gene axis, with paired differences
between the two axes.

Usage: eval_r0.py --run-dir DIR --out-dir reports/softprior_multitask_r1/tables
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_data as R  # noqa: E402
import r1_eval as E  # noqa: E402


def track_a_oof(data: R.Data, name: str) -> np.ndarray:
    d = os.path.join(R.REPO, "reports", "pathway_sensitivity_a1", "oof", "a2_main_d284715", "oof")
    files = sorted(glob.glob(os.path.join(d, f"{name}__rep*.tsv.gz")))
    assert len(files) == 3
    mi = {m: i for i, m in enumerate(data.models)}
    di = {x: j for j, x in enumerate(data.drugs)}
    pred = np.full((3,) + data.Y.shape, np.nan)
    for f in files:
        t = pd.read_csv(f, sep="\t")
        pred[t.repeat.values, t.model_id.map(mi).values, t.drug_id.map(di).values] = t.pred.values
    assert np.isfinite(pred).all()
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    data = R.load_data()
    sc = E.Scorer(data)
    os.makedirs(args.out_dir, exist_ok=True)
    committed = pd.read_csv(os.path.join(R.REPO, "reports", "pathway_sensitivity_a1", "tables", "internal_summary.tsv"),
                            sep="\t").set_index("model")
    rows, diffs, check = [], [], {}
    for kind in ("A0", "A1", "A1r", "A3"):
        old = track_a_oof(data, kind)
        new = E.load_oof(args.run_dir, kind)["pred"]
        ro = sc.summary_row(f"{kind}_trackA_axis_committed_oof", old)
        rn = sc.summary_row(f"{kind}_fixedID_axis", new)
        check[kind] = dict(committed=float(committed.loc[kind, "macro_drug_normalized_rmse"]),
                           rescored_by_r1=ro["macro_drug_normalized_rmse"],
                           committed_ci=[float(committed.loc[kind, "ci_low"]), float(committed.loc[kind, "ci_high"])],
                           rescored_ci=[ro["ci_low"], ro["ci_high"]])
        assert abs(check[kind]["committed"] - check[kind]["rescored_by_r1"]) < 5e-6, check[kind]
        rows += [ro, rn]
        d = sc.difference_row(f"{kind}_fixedID_axis", f"{kind}_trackA_axis_committed_oof", sc.cube(new), sc.cube(old))
        d["max_abs_prediction_change"] = float(np.abs(new - old).max())
        diffs.append(d)
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "r0_track_a_models_fixed_axis.tsv"), sep="\t", index=False,
                              float_format="%.6g")
    pd.DataFrame(diffs).to_csv(os.path.join(args.out_dir, "r0_axis_change_differences.tsv"), sep="\t", index=False,
                               float_format="%.6g")
    with open(os.path.join(args.out_dir, "r0_scorer_check.json"), "w") as fh:
        json.dump(check, fh, indent=1)
    print(pd.DataFrame(rows).to_string())
    print(pd.DataFrame(diffs).to_string())


if __name__ == "__main__":
    main()
