"""R6-R: the two-block Ridge of R6 (Z0) fitted on the SAME fixed view cache as S / SD / M.

    delta = b + W_c * standardised mean log-normalised RNA of the view + W_d * ECFP4 bits
Every fitting condition appears once per stored view of its pool (24 views: 8 per keep-rate), i.e.
exactly the finite set of views the networks draw from. It is a finite-view linear model and is not
described as equivalent to a network that resamples views.

The design matrix has one row per (condition, view); its normal equations are accumulated from the
block structure (control block depends on the view only, drug block on the condition only), so no
row is materialised. Centring and the unit ridge on the alpha-scaled blocks follow the original
`ridge.solve`.

Penalties (alpha_c, alpha_d): chosen per run on the fold's INNER development drugs (fit on the inner
fit drugs, primary metric at p = 0.25, thinning seed 0), then the model is refitted on all outer-fit
drugs. The held fold is never used for the choice. `--alphas` fixes them (final setting).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from src.depth_robust_r6 import data as D
from src.depth_robust_r6.ridge import ALPHA_C, ALPHA_D, eval_feats

from . import common as C
from .metrics import cond_rmse, drug_macro
from .views import load_views, view_means


class Normal:
    """Centred normal equations of the (condition x view) design for one set of fitting drugs."""

    def __init__(self, d, fit_drugs, Xv, nv):
        tt = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64)
        Lf, Df = d.conditions(fit_drugs)
        pf = C.pool_of(d, Lf, Df)
        n_pool = len(d.pools)
        Xv_t = tt(Xv)                                                 # [n_pool * nv, G], views of a pool contiguous
        Sx = Xv_t.reshape(n_pool, nv, -1).sum(1)                      # per-pool sum over its views
        npc = torch.as_tensor(np.bincount(pf, minlength=n_pool), dtype=torch.float64)
        w = npc.repeat_interleave(nv)
        F, Y, Sc = tt(d.fp[Df]), tt(d.Y[Lf, Df]), Sx[torch.as_tensor(pf)]
        n = len(Lf) * nv
        Gc = Xv_t.shape[1]
        XtX = torch.empty(Gc + F.shape[1], Gc + F.shape[1], dtype=torch.float64)
        XtX[:Gc, :Gc] = (Xv_t * w[:, None]).T @ Xv_t
        XtX[:Gc, Gc:] = Sc.T @ F
        XtX[Gc:, :Gc] = XtX[:Gc, Gc:].T
        XtX[Gc:, Gc:] = nv * (F.T @ F)
        XtY = torch.cat([Sc.T @ Y, nv * (F.T @ Y)], 0)
        mu = torch.cat([Sc.sum(0), nv * F.sum(0)]) / n
        ym = Y.mean(0)
        self.XtX = XtX - n * torch.outer(mu, mu)
        self.XtY = XtY - n * torch.outer(mu, ym)
        self.mu, self.ym, self.Gc, self.n_rows, self.n_cond = mu, ym, Gc, n, len(Lf)

    def solve(self, ac: float, ad: float):
        s = torch.cat([torch.full((self.Gc,), ac ** -0.5, dtype=torch.float64),
                       torch.full((self.XtX.shape[0] - self.Gc,), ad ** -0.5, dtype=torch.float64)])
        A = self.XtX * s[:, None] * s[None, :] + torch.eye(len(s), dtype=torch.float64)
        W = torch.linalg.solve(A, self.XtY * s[:, None]) * s[:, None]  # weights on the unscaled features
        return W, self.ym - self.mu @ W


def predict(W, b, Zc, Fd):
    X = torch.cat([torch.as_tensor(Zc, dtype=torch.float64), torch.as_tensor(Fd, dtype=torch.float64)], 1)
    return (X @ W + b).float().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True, choices=["source", "plate", "final"])
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--seed", type=int, default=42, help="seed of the view cache")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--alphas", default=None, help="json [alpha_c, alpha_d]; required in the final setting")
    a = ap.parse_args()
    if a.setting == "final":
        fm = Path("manifests/r6_followup_v1/freeze_manifest.json")
        assert fm.exists() and a.alphas, "freeze the follow-up configuration (with fixed penalties) before any final-setting run"
    run = C.assets() / "runs" / a.stage / a.run_id
    run.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    d = C.load()
    v = load_views(a.seed)
    nv = int(v["reps"]) * len(D.PS)
    std = lambda m: (m - d.x_mean) / d.x_sd
    Xv = std(view_means(v))
    fit, held = C.split_spec(d, a.setting, a.fold)
    rec = dict(run_id=a.run_id, stage=a.stage, arm="R", setting=a.setting, fold=a.fold, seed=a.seed, views_per_pool=nv,
               view_cache_sha256_counts=D.sha256_array(v["cnt"]), n_fit_drugs=len(fit), n_eval_drugs=len(held))
    if a.alphas:
        ac, ad = json.loads(a.alphas)
        rec["alphas"] = dict(alpha_c=ac, alpha_d=ad, source="fixed by the caller")
    else:
        ifit, idev = C.split_spec(d, a.setting, a.fold, True)
        nm = Normal(d, ifit, Xv, nv)
        Li, Di = d.conditions(idev)
        if a.setting == "plate":                                       # inner held plate: score its own conditions only
            q = [x for x in sorted(int(p_) for p_ in d.pools.plate_index.unique()) if x != a.fold][0]
            Li, Di, _ = C.plate_conditions(d, idev, q)
        pi = C.pool_of(d, Li, Di)
        ev = eval_feats(d)
        Ti = d.Y[Li, Di]
        grid = []
        for ac_ in ALPHA_C:
            for ad_ in ALPHA_D:
                W, b = nm.solve(ac_, ad_)
                sc = {k: drug_macro(cond_rmse(predict(W, b, std(ev[k][pi]), d.fp[Di]), Ti), Di) for k in [(1.0, 0), (0.25, 0)]}
                grid.append(dict(alpha_c=ac_, alpha_d=ad_, p1=sc[(1.0, 0)], p025=sc[(0.25, 0)]))
                print(json.dumps(grid[-1]), flush=True)
        best = min(grid, key=lambda g: g["p025"])
        ac, ad = best["alpha_c"], best["alpha_d"]
        rec["grid"] = grid
        rec["alphas"] = dict(alpha_c=ac, alpha_d=ad, source="inner development drugs of the fold, primary metric at p=0.25, thinning seed 0",
                             n_inner_fit_drugs=len(ifit), n_inner_dev_drugs=len(idev),
                             at_edge=bool(ac in (ALPHA_C[0], ALPHA_C[-1]) or ad in (ALPHA_D[0], ALPHA_D[-1])))
    nm = Normal(d, fit, Xv, nv)
    W, b = nm.solve(ac, ad)
    rec.update(n_fit_conditions=nm.n_cond, design_rows=nm.n_rows, n_features=int(W.shape[0]),
               drug_block_weight_rms=float((W[nm.Gc:] ** 2).mean().sqrt()), control_block_weight_rms=float((W[:nm.Gc] ** 2).mean().sqrt()))
    np.savez(run / "model.npz", W=W.float().numpy(), b=b.float().numpy(), alpha_c=ac, alpha_d=ad, fit_drugs=np.asarray(fit))
    if a.setting in ("source", "plate"):
        L, Dg = d.conditions(held)
        pq = C.pool_of(d, L, Dg)
        ev = eval_feats(d)
        pr = {k: predict(W, b, std(ev[k][pq]), d.fp[Dg]) for k in C.VIEWS}
        np.savez_compressed(run / "pred.npz", line=L, drug=Dg, **{f"p{p}_s{s}": x.astype(np.float16) for (p, s), x in pr.items()})
        T = d.Y[L, Dg]
        rec["eval_role"] = "held_fold"
        rec["scores"] = {f"p{p}_s{s}": drug_macro(cond_rmse(x, T), Dg) for (p, s), x in pr.items()}
        rec["pred_sha256"] = D.sha256_file(run / "pred.npz")
    rec["seconds"] = round(time.time() - t0, 1)
    rec["commit"] = C.git_commit(); rec["job"] = os.environ.get("SLURM_JOB_ID")
    C.dump_json(rec, run / "record.json")
    print(json.dumps({k: x for k, x in rec.items() if k != "grid"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
