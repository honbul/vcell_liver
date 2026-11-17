#!/usr/bin/env python3
"""Public comparator: cwKBMF (component-wise kernelized Bayesian matrix factorization), authors' MATLAB
code run unmodified under GNU Octave.

What comes from the publication / archive: the variational training and prediction functions, the
hyper-parameter values (alpha, beta) = (1, 1) for both gamma priors, (sigma_g, sigma_h, sigma_y) =
(0.1, 0.1, 1), and R scanned over a small set.
What is this study's choice (the archive has no data pipeline): one Gaussian kernel per Hallmark set on the
standardised rank RNA of its genes plus one kernel for all genes outside every set,
k(a, b) = exp(-||a - b||^2 / d) with d the number of genes in the view; identity kernel on the drug side;
responses standardised per drug with training-only mean/SD; 200 variational iterations; R chosen from
{10, 20} by the same inner family folds and inner loss as every other model (one R for all drugs).
The code has no sample weights, so fits are unweighted; scalers use the family weights like all models.

Usage: run_cwkbmf.py --run-dir DIR --sets internal|external [--task-index I --n-tasks N] [--iteration 200]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy.io import loadmat, savemat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_data as R  # noqa: E402
import r1_solvers as S  # noqa: E402
from r1_data import a1core  # noqa: E402
from run_grid import training_sets  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RS = [10, 20]
PARAMS = dict(alpha_lambda=1.0, beta_lambda=1.0, alpha_eta=1.0, beta_eta=1.0, sigmag=0.1, sigmah=0.1, sigmay=1.0,
              seed=1606)


def views(sets: list, G: int) -> list:
    rest = np.setdiff1d(np.arange(G), np.unique(np.concatenate(sets)))
    return list(sets) + [rest]


def kernels(Xa: np.ndarray, Xb: np.ndarray, vws: list) -> np.ndarray:
    """[rows of a, rows of b, view] Gaussian kernels; width = number of (non-constant) genes in the view."""
    K = np.empty((Xa.shape[0], Xb.shape[0], len(vws)))
    for m, idx in enumerate(vws):
        A, B = Xa[:, idx], Xb[:, idx]
        d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
        K[:, :, m] = np.exp(-np.clip(d2, 0, None) / len(idx))
    return K


def prepare(X_tr, Y_tr, fam_tr, X_te, vws):
    wt = S.norm_weights(a1core.family_weights(fam_tr))
    mu, sd = S.wmean_sd(X_tr, wt)
    Xs, Xt = S.standardise(X_tr, mu, sd), S.standardise(X_te, mu, sd)
    ymu, ysd = S.wmean_sd(Y_tr, wt)
    ysd = np.where(ysd < 1e-8, 1.0, ysd)
    return kernels(Xs, Xs, vws), kernels(Xs, Xt, vws), (Y_tr - ymu) / ysd, ymu, ysd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--sets", required=True, choices=["internal", "external"])
    ap.add_argument("--task-index", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--iteration", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first units only")
    args = ap.parse_args()
    cfg = R.load_config()
    code = os.path.join(R.rpath(cfg["paths"]["assets"]), "external", "cwkbmf", "src", "cwkbmf")
    assert os.path.exists(os.path.join(code, "cwkbmf_1mkl1k_regression_variational_train.m")), code
    data = R.load_data(cfg)
    ext = R.load_external(cfg, data) if args.sets == "external" else None
    units = training_sets(cfg, data, args.sets, ext)
    units = [u for i, u in enumerate(units) if i % args.n_tasks == args.task_index][:args.limit]
    vws = views(data.sets, len(data.genes))
    out_dir = os.path.join(args.run_dir, "cmp", "cwkbmf")
    os.makedirs(out_dir, exist_ok=True)
    n_inner = cfg["split"]["inner_folds"]
    for sid, tr, X_te, te_models, seed in units:
        out = os.path.join(out_dir, f"{sid}.npz")
        if os.path.exists(out):
            raise FileExistsError(out)
        t0 = time.time()
        X, Y, fam = data.rank[tr], data.Y[tr], data.fam[tr]
        inner = a1core.group_folds(fam, n_inner, seed)
        mat = dict(n_fits=n_inner + 1, Rs=np.array(RS, float), iteration=float(args.iteration), **PARAMS)
        scal = []
        for k in range(n_inner):
            itr, ite = inner != k, inner == k
            Ktr, Kte, Z, ymu, ysd = prepare(X[itr], Y[itr], fam[itr], X[ite], vws)
            mat[f"Ktr_{k + 1}"], mat[f"Kte_{k + 1}"], mat[f"Z_{k + 1}"] = Ktr, Kte, Z
            scal.append((ymu, ysd))
        Ktr, Kte, Z, ymu, ysd = prepare(X, Y, fam, X_te, vws)
        mat[f"Ktr_{n_inner + 1}"], mat[f"Kte_{n_inner + 1}"], mat[f"Z_{n_inner + 1}"] = Ktr, Kte, Z
        with tempfile.TemporaryDirectory(dir=out_dir) as td:
            fin, fout = os.path.join(td, "in.mat"), os.path.join(td, "out.mat")
            savemat(fin, mat)
            cmd = f"cwkbmf_run('{code}', '{fin}', '{fout}')"
            res = subprocess.run(["octave", "--no-gui", "--quiet", "--path", HERE, "--eval", cmd], capture_output=True,
                                 text=True, cwd=td)
            if res.returncode != 0 or not os.path.exists(fout):
                raise RuntimeError(f"octave failed for {sid}: {res.stderr[-2000:]}")
            o = loadmat(fout)
        loss = np.zeros((len(RS), Y.shape[1]))
        n_tot = 0
        for k in range(n_inner):
            ite = inner == k
            Zte = (Y[ite] - scal[k][0]) / scal[k][1]
            f_te, inv = np.unique(fam[ite], return_inverse=True)
            for j, r in enumerate(RS):
                P = o[f"P_{k + 1}_R{r}"]
                assert P.shape == Zte.shape and np.isfinite(P).all()
                agg = np.zeros((len(f_te), Y.shape[1]))
                np.add.at(agg, inv, (P - Zte) ** 2)
                loss[j] += (agg / np.bincount(inv)[:, None]).sum(axis=0)
            n_tot += len(f_te)
        loss /= n_tot
        score = np.sqrt(loss).mean(axis=1)
        best = int(np.argmin(score))                               # ties -> smaller R
        pred = o[f"P_{n_inner + 1}_R{RS[best]}"] * ysd + ymu
        assert pred.shape == (len(te_models), Y.shape[1]) and np.isfinite(pred).all()
        meta = dict(set=sid, seconds=round(time.time() - t0, 1), R_chosen=int(RS[best]), iteration=args.iteration,
                    n_views=len(vws), inner_score={str(r): float(s) for r, s in zip(RS, score)})
        np.savez(out[:-4] + ".tmp.npz", pred=pred, models=np.array(te_models), inner_loss=loss, R=np.array(RS),
                 view_weight_mean_abs=np.abs(o[f"e_{n_inner + 1}_R{RS[best]}"]).mean(axis=1), meta=json.dumps(meta))
        os.replace(out[:-4] + ".tmp.npz", out)
        print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
