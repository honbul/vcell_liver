#!/usr/bin/env python3
"""Optional public comparator: TabPFN regressor on a training-only PCA of the rank RNA, one model per drug.

Fixed in advance, no tuning and no backend choice after seeing results: TabPFN v2 regressor weights
(the newer default weights need an account registration, which this study does not do), default
settings, CPU, random_state 0; features = first min(50, n_train - 1) principal components of the
standardised rank RNA (scaler and PCA fitted on the training rows only); raw AA as the target.
Because nothing is tuned there is no inner loop.

Usage: run_tabpfn.py --run-dir DIR --sets internal|external [--task-index I --n-tasks N] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_data as R  # noqa: E402
import r1_solvers as S  # noqa: E402
from r1_data import a1core  # noqa: E402
from run_grid import training_sets  # noqa: E402

MAX_DIM = 50


def pca_features(X_tr, fam_tr, X_te):
    wt = S.norm_weights(a1core.family_weights(fam_tr))
    mu, sd = S.wmean_sd(X_tr, wt)
    Xs, Xt = S.standardise(X_tr, mu, sd), S.standardise(X_te, mu, sd)
    k = min(MAX_DIM, X_tr.shape[0] - 1)
    U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
    return Xs @ Vt[:k].T, Xt @ Vt[:k].T, k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--sets", required=True, choices=["internal", "external"])
    ap.add_argument("--task-index", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first drugs of the first unit only")
    args = ap.parse_args()
    cfg = R.load_config()
    os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", os.path.join(R.rpath(cfg["paths"]["assets"]), "external", "tabpfn_cache"))
    import tabpfn
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    data = R.load_data(cfg)
    ext = R.load_external(cfg, data) if args.sets == "external" else None
    units = training_sets(cfg, data, args.sets, ext)
    units = [u for i, u in enumerate(units) if i % args.n_tasks == args.task_index]
    if args.sets == "internal":
        dcols = np.arange(len(data.drugs))
    else:
        dcols = np.array([data.drugs.index(x) for x in sorted(ext.drug_match.limore_name)])   # independent per drug
    if args.limit:
        units, dcols = units[:1], dcols[:args.limit]
    out_dir = os.path.join(args.run_dir, "cmp", "tabpfn")
    os.makedirs(out_dir, exist_ok=True)
    for sid, tr, X_te, te_models, _ in units:
        out = os.path.join(out_dir, f"{sid}.npz")
        if os.path.exists(out):
            raise FileExistsError(out)
        t0 = time.time()
        F_tr, F_te, k = pca_features(data.rank[tr], data.fam[tr], X_te)
        pred = np.full((len(te_models), len(data.drugs)), np.nan)
        for d in dcols:
            m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0)
            m.fit(F_tr, data.Y[tr][:, d])
            pred[:, d] = m.predict(F_te)
        meta = dict(set=sid, seconds=round(time.time() - t0, 1), pca_dim=int(k), tabpfn=tabpfn.__version__,
                    weights="v2 regressor", n_drugs=int(len(dcols)))
        np.savez(out[:-4] + ".tmp.npz", pred=pred, models=np.array(te_models), meta=json.dumps(meta))
        os.replace(out[:-4] + ".tmp.npz", out)
        print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
