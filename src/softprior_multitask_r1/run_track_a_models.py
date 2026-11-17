#!/usr/bin/env python3
"""Stage R1-0 (part 2): rerun Track A's own predictors on the fixed-ID gene axis.

A0 (training mean), A1 (log2 FPKM Ridge), A1r (rank Ridge) and A3 (Hallmark singscore Ridge) are fitted
with Track A's `nested.fit_predict` (sklearn StandardScaler + Ridge, Track A alpha grid, per-drug alpha)
on Track A's outer folds. Only the gene axis differs from Track A.

Usage: run_track_a_models.py --run-dir DIR [--kinds A0,A1,A1r,A3]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_data as R  # noqa: E402
import r1_eval as E  # noqa: E402
import nested as a1nested  # noqa: E402  (Track A)

A1_ALPHAS = [0.01, 0.1, 1, 10, 100, 1000, 10000]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--kinds", default="A0,A1,A1r,A3")
    args = ap.parse_args()
    cfg = R.load_config()
    data = R.load_data(cfg)
    feats = {"A0": np.zeros((len(data.models), 1)), "A1": data.expr, "A1r": data.rank, "A3": data.hallmark}
    reps = sorted(data.splits.repeat.unique())
    for kind in args.kinds.split(","):
        t0 = time.time()
        X = feats[kind]
        pred = np.full((len(reps),) + data.Y.shape, np.nan)
        alpha = np.full((len(reps),) + data.Y.shape, np.nan)
        for r in reps:
            fold = data.outer_fold(r)
            for k, tr, te, p, info in a1nested.outer_folds(
                    X, data.Y, data.fam, fold, kind, A1_ALPHAS, cfg["split"]["inner_folds"],
                    lambda k, r=r: R.inner_seed(cfg, r, k), len(data.sets)):
                pred[r, te] = p
                alpha[r, te] = info["alpha"]
        E.save_oof(args.run_dir, kind, pred, data.models, data.drugs, alpha=alpha)
        print(f"{kind}: {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
