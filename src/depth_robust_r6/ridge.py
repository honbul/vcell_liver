"""Z0 and the trivial references: no network, same cells, same views, same truth.

    nochange   delta = 0
    linemean   per-cell-line mean response over the fitting drugs (needs the line's identity only)
    Z0         two-block ridge:  delta = b + W_c * standardised mean log-normalised control set
                                           + W_d * ECFP4 bits
               fitted on the fitting drugs with the SAME count-thinning augmentation as the
               networks: every fitting condition appears once per keep-rate p in {1, 0.5, 0.25},
               the control set being the first 64 fit_input cells of its pool thinned with a fixed
               seed. Penalties (alpha_c, alpha_d) are chosen on the dev fold at p = 0.25 and reused
               unchanged in the final setting.
    Z0_p1      diagnostic only (R6-0): the same ridge fitted on full-observation sets alone

Evaluation uses the fixed eval_input cells and the stored evaluation views, like every other model.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .count_view import lognorm, thin_np
from .evaluate import cond_rmse, drug_macro

ALPHA_C = [10.0, 100.0, 1000.0, 10000.0]
ALPHA_D = [30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0]
# The first grid stopped at alpha_d = 3000 and the choice sat on that edge (run Z0_dev_v1). The grid
# was extended ONCE upward (the plan's single range extension for a simple baseline); see DECISIONS.md.
VIEWS = [(1.0, 0)] + [(p, s) for p in D.PS[1:] for s in D.THIN_SEEDS]


def set_mean(c: np.ndarray, r: np.ndarray, n_sets: int, n: int = D.EVAL_N, take: int | None = None) -> np.ndarray:
    """Mean log-normalised profile of each set of n cells (optionally only its first `take` cells)."""
    x = lognorm(c.astype(np.float64), (c.sum(1) + r).astype(np.float64)).reshape(n_sets, n, -1)
    return x[:, :take].mean(1) if take else x.mean(1)


def fit_views(d: D.R6Data, ps) -> dict:
    """Per-pool control feature of the fitting views: first 64 fit_input cells, thinned with seed 100."""
    rows = np.concatenate([np.arange(r.fit_start, r.fit_start + D.EVAL_N) for r in d.pools.itertuples()])
    out = {}
    for p in ps:
        rng = np.random.Generator(np.random.PCG64([20261117, int(round(p * 1000)), 100]))
        c, r = thin_np(d.cnt[rows], d.crest[rows], p, rng)
        out[p] = set_mean(c, r, len(d.pools))
    return out


def eval_feats(d: D.R6Data, take: int | None = None) -> dict:
    rows = np.concatenate([np.arange(r.eval_start, r.eval_end) for r in d.pools.itertuples()])
    out = {(1.0, 0): set_mean(d.cnt[rows], d.crest[rows], len(d.pools), take=take)}
    for k, (c, r) in d.eval_views.items():
        out[k] = set_mean(c, r, len(d.pools), take=take)
    return out


def solve(Xc, Xd, Y, ac, ad, dev):
    """Primal two-block ridge on centred inputs; returns a predictor closure."""
    X = torch.cat([Xc / ac ** 0.5, Xd / ad ** 0.5], 1)
    mu, ym = X.mean(0), Y.mean(0)
    Xz = X - mu
    A = Xz.T @ Xz + torch.eye(X.shape[1], dtype=X.dtype, device=dev)
    W = torch.linalg.solve(A, Xz.T @ (Y - ym))
    return lambda Zc, Zd: (torch.cat([Zc / ac ** 0.5, Zd / ad ** 0.5], 1) - mu) @ W + ym


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True, choices=["dev", "final"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--alphas", default=None, help="json [alpha_c, alpha_d]; required in the final setting")
    ap.add_argument("--diagnostic", action="store_true", help="also fit Z0_p1 and score N=16 sets (dev only)")
    a = ap.parse_args()
    run = Path(os.environ["R6_ASSETS"]) / "runs" / a.setting / a.run_id
    run.mkdir(parents=True, exist_ok=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    d = D.load()
    fit_drugs, eval_drugs = d.split_spec(a.setting)
    if a.setting == "final":
        assert a.alphas, "the final setting reuses the penalties chosen in development"
        assert Path("manifests/depth_robust_r6/freeze_manifest.json").exists(), "freeze before the held-out evaluation"
    Lf, Df = d.conditions(fit_drugs); Le, De = d.conditions(eval_drugs)
    pf, pe = d.pool_id[d.plate[Lf, Df], Lf], d.pool_id[d.plate[Le, De], Le]
    T = d.Y[Le, De]
    std = lambda m: (m - d.x_mean) / d.x_sd
    tt = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64, device=dev)
    fv = fit_views(d, D.PS)
    ev = eval_feats(d)
    Yf = tt(d.Y[Lf, Df]); Fd_f, Fd_e = tt(d.fp[Df]), tt(d.fp[De])

    def fit_predict(ps, ac, ad, feats=ev):
        Xc = torch.cat([tt(std(fv[p][pf])) for p in ps]); Xd = Fd_f.repeat(len(ps), 1); Y = Yf.repeat(len(ps), 1)
        f = solve(Xc, Xd, Y, ac, ad, dev)
        return {k: f(tt(std(v[pe])), Fd_e).float().cpu().numpy() for k, v in feats.items()}

    score = lambda pr, k: drug_macro(cond_rmse(pr[k], T), De)
    rec = dict(run_id=a.run_id, setting=a.setting, n_fit_conditions=int(len(Lf)), n_eval_conditions=int(len(Le)),
               n_fit_drugs=len(fit_drugs), n_eval_drugs=len(eval_drugs))
    preds = {}
    # trivial references
    lm = np.zeros((d.Y.shape[0], d.Y.shape[2]))
    for l in np.unique(Lf):
        lm[l] = d.Y[Lf[Lf == l], Df[Lf == l]].mean(0)
    preds["nochange"] = {k: np.zeros_like(T) for k in VIEWS}
    preds["linemean"] = {k: lm[Le].astype(np.float32) for k in VIEWS}
    if a.alphas:
        ac, ad = json.loads(a.alphas)
        rec["alphas"] = dict(alpha_c=ac, alpha_d=ad, source="fixed by the caller (development choice)")
    else:
        grid = []
        for ac_ in ALPHA_C:
            for ad_ in ALPHA_D:
                pr = fit_predict(D.PS, ac_, ad_, {k: ev[k] for k in [(1.0, 0), (0.25, 0)]})
                grid.append(dict(alpha_c=ac_, alpha_d=ad_, p1=score(pr, (1.0, 0)), p025=score(pr, (0.25, 0))))
                print(json.dumps(grid[-1]), flush=True)
        best = min(grid, key=lambda g: g["p025"])
        ac, ad = best["alpha_c"], best["alpha_d"]
        rec["grid"] = grid
        rec["alphas"] = dict(alpha_c=ac, alpha_d=ad, source="dev fold, primary metric at p=0.25, thinning seed 0",
                             at_edge=bool(ac in (ALPHA_C[0], ALPHA_C[-1]) or ad in (ALPHA_D[0], ALPHA_D[-1])))
    preds["Z0"] = fit_predict(D.PS, ac, ad)
    if a.diagnostic and a.setting == "dev":
        preds["Z0_p1"] = fit_predict((1.0,), ac, ad)
        ev16 = eval_feats(d, take=16)
        rec["n16"] = {m: {f"p{k[0]}_s{k[1]}": drug_macro(cond_rmse(v, T), De) for k, v in fit_predict(ps, ac, ad, ev16).items()}
                      for m, ps in [("Z0", D.PS), ("Z0_p1", (1.0,))]}
    rec["scores"] = {m: {f"p{k[0]}_s{k[1]}": score(pr, k) for k in VIEWS} for m, pr in preds.items()}
    for m, pr in preds.items():
        np.savez_compressed(run / f"pred_{a.setting}_{m}.npz", line=Le, drug=De,
                            **{f"p{k[0]}_s{k[1]}": v.astype(np.float16) for k, v in pr.items()})
    rec["seconds"] = round(time.time() - t0, 1)
    rec["commit"] = os.popen("git rev-parse HEAD").read().strip()
    D.dump_json(rec, run / "record.json")
    print(json.dumps({k: v for k, v in rec.items() if k != "grid"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
