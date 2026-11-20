#!/usr/bin/env python3
"""Track A's own predictors (A1r rank Ridge, A3 Hallmark Ridge, A1 log2 Ridge) on the fixed-ID axis for
the LCCL external sets, with Track A's nested routine and alpha grid. Predictions only; no LCCL label.

Usage: run_track_a_external.py --run-dir DIR
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_data as R  # noqa: E402
import nested as a1nested  # noqa: E402  (Track A)
from run_grid import training_sets  # noqa: E402
from run_track_a_models import A1_ALPHAS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    cfg = R.load_config()
    data = R.load_data(cfg)
    ext = R.load_external(cfg, data)
    row = {m: i for i, m in enumerate(ext.models)}
    src = {"A1": (data.expr, ext.expr), "A1r": (data.rank, ext.rank), "A3": (data.hallmark, ext.hallmark)}
    for kind, (Xs, Xt_all) in src.items():
        out_dir = os.path.join(args.run_dir, "cmp", kind)
        os.makedirs(out_dir, exist_ok=True)
        for sid, tr, _, te_models, seed in training_sets(cfg, data, "external", ext):
            out = os.path.join(out_dir, f"{sid}.npz")
            if os.path.exists(out):
                raise FileExistsError(out)
            Xt = Xt_all[[row[m] for m in te_models]]
            pred, info = a1nested.fit_predict(Xs[tr], data.Y[tr], data.fam[tr], Xt, kind, A1_ALPHAS,
                                              cfg["split"]["inner_folds"], seed, len(data.sets))
            np.savez(out[:-4] + ".tmp.npz", pred=pred, models=np.array(te_models), alpha=info["alpha"])
            os.replace(out[:-4] + ".tmp.npz", out)
        print(kind, "done", flush=True)


if __name__ == "__main__":
    main()
