#!/usr/bin/env python3
"""Baseline: raw-RNA (log2 FPKM) ElasticNet, one model per drug, tuned inside the training families.

Features and labels are standardised with training-only family-weighted mean/SD. For each drug the
(alpha, l1_ratio) pair is chosen by the same inner 3-fold family split and the same inner loss as every
other model; ties go to the larger alpha, then the larger l1_ratio. Fits use family weights.

Usage: run_enet.py --run-dir DIR --sets internal|external [--task-index I --n-tasks N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_data as R  # noqa: E402
import r1_solvers as S  # noqa: E402
from r1_data import a1core  # noqa: E402
from run_grid import training_sets  # noqa: E402


def fit_path(Xs, Z, wt, Xt, alphas, l1_ratios, max_iter, tol):
    """Predictions [l1_ratio, alpha, row, drug] in standardised label units; warm start down the alpha path."""
    out = np.zeros((len(l1_ratios), len(alphas), Xt.shape[0], Z.shape[1]))
    n_warn = 0
    sw = wt * len(wt)
    for a, l1 in enumerate(l1_ratios):
        for d in range(Z.shape[1]):
            model = ElasticNet(alpha=1.0, l1_ratio=l1, fit_intercept=False, warm_start=True, max_iter=max_iter, tol=tol)
            for b in np.argsort(alphas)[::-1]:
                model.set_params(alpha=float(alphas[b]))
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(Xs, Z[:, d], sample_weight=sw)
                    n_warn += sum(issubclass(x.category, ConvergenceWarning) for x in w)
                out[a, b, :, d] = Xt @ model.coef_
    return out, n_warn


def scale(X_tr, Y_tr, fam_tr, X_te):
    wt = S.norm_weights(a1core.family_weights(fam_tr))
    mu, sd = S.wmean_sd(X_tr, wt)
    ymu, ysd = S.wmean_sd(Y_tr, wt)
    ysd = np.where(ysd < 1e-8, 1.0, ysd)
    return wt, S.standardise(X_tr, mu, sd), S.standardise(X_te, mu, sd), (Y_tr - ymu) / ysd, ymu, ysd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--sets", required=True, choices=["internal", "external"])
    ap.add_argument("--task-index", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    args = ap.parse_args()
    cfg = R.load_config()
    en = cfg["baselines"]["enet"]
    alphas, l1s = np.array(en["alpha"], float), list(en["l1_ratio"])
    data = R.load_data(cfg)
    ext = R.load_external(cfg, data) if args.sets == "external" else None
    units = training_sets(cfg, data, args.sets, ext)
    # same row selection as run_grid, but on log2 expression
    if args.sets == "internal":
        feats = {sid: data.expr[[data.models.index(m) for m in te]] for sid, _, _, te, _ in units}
        dcols = np.arange(len(data.drugs))
    else:
        feats = {sid: ext.expr[[ext.models.index(m) for m in te]] for sid, _, _, te, _ in units}
        dcols = np.array([data.drugs.index(x) for x in sorted(ext.drug_match.limore_name)])   # drugs are independent here
    units = [u for i, u in enumerate(units) if i % args.n_tasks == args.task_index]
    out_dir = os.path.join(args.run_dir, "cmp", "enet")
    os.makedirs(out_dir, exist_ok=True)
    n_inner = cfg["split"]["inner_folds"]
    for sid, tr, _, te_models, seed in units:
        out = os.path.join(out_dir, f"{sid}.npz")
        if os.path.exists(out):
            raise FileExistsError(out)
        t0 = time.time()
        X, Y, fam = data.expr[tr], data.Y[tr][:, dcols], data.fam[tr]
        inner = a1core.group_folds(fam, n_inner, seed)
        loss = np.zeros((len(l1s), len(alphas), Y.shape[1]))
        n_warn, n_tot = 0, 0
        for k in range(n_inner):
            itr, ite = inner != k, inner == k
            wt, Xs, Xt, Z, ymu, ysd = scale(X[itr], Y[itr], fam[itr], X[ite])
            P, w = fit_path(Xs, Z, wt, Xt, alphas, l1s, en["max_iter"], en["tol"])
            n_warn += w
            se = (P - (Y[ite] - ymu) / ysd) ** 2
            f_te, inv = np.unique(fam[ite], return_inverse=True)
            agg = np.zeros(se.shape[:2] + (len(f_te), Y.shape[1]))
            np.add.at(agg, (slice(None), slice(None), inv), se)
            loss += (agg / np.bincount(inv)[:, None]).sum(axis=2)
            n_tot += len(f_te)
        loss /= n_tot
        # candidates ordered weakest -> strongest: alpha ascending, then l1_ratio ascending; ties -> last
        order = [(a, b) for b in np.argsort(alphas) for a in range(len(l1s))]
        flat = np.array([loss[a, b] for a, b in order])
        best = len(order) - 1 - np.argmin(flat[::-1], axis=0)
        wt, Xs, Xt, Z, ymu, ysd = scale(X, Y, fam, feats[sid])
        P, w = fit_path(Xs, Z, wt, Xt, alphas, l1s, en["max_iter"], en["tol"])
        n_warn += w
        pred = np.full((len(te_models), len(data.drugs)), np.nan)
        ch_alpha, ch_l1 = np.full(len(data.drugs), np.nan), np.full(len(data.drugs), np.nan)
        for j, d in enumerate(dcols):
            a, b = order[best[j]]
            pred[:, d] = P[a, b, :, j] * ysd[j] + ymu[j]
            ch_alpha[d], ch_l1[d] = alphas[b], l1s[a]
        meta = dict(set=sid, seconds=round(time.time() - t0, 1), n_convergence_warnings=int(n_warn),
                    n_fits=int(len(l1s) * len(alphas) * Y.shape[1] * (n_inner + 1)))
        np.savez(out[:-4] + ".tmp.npz", pred=pred, models=np.array(te_models), alpha=ch_alpha, l1_ratio=ch_l1,
                 inner_loss=loss, meta=json.dumps(meta))
        os.replace(out[:-4] + ".tmp.npz", out)
        print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
