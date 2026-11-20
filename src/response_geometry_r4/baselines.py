"""Closed-form baselines fitted on training-drug units; alpha chosen on validation units."""
import json, os
import numpy as np
import pandas as pd

from .common import public, dump
from .data import Store, TrainStats
from .evaluate import unit_metrics, summarise

ALPHAS = [1, 10, 100, 1000, 10000]


class WRidge:
    """Cell-balanced weighted ridge with intercept; one eigendecomposition for the whole alpha grid."""

    def __init__(self, X, Y, w):
        w = w / w.sum()
        self.xm = (w[:, None] * X).sum(0); self.ym = (w[:, None] * Y).sum(0)
        Xc = X - self.xm; Yc = Y - self.ym
        G = Xc.T @ (w[:, None] * Xc)
        self.lam, self.V = np.linalg.eigh(G)
        self.VtB = self.V.T @ (Xc.T @ (w[:, None] * Yc))

    def coef(self, alpha_scaled):
        return self.V @ (self.VtB / (self.lam + alpha_scaled)[:, None])

    def predict(self, X, beta):
        return (X - self.xm) @ beta + self.ym


def cell_weights(S, idx):
    c = S.cell_idx[idx]
    n = np.bincount(c, minlength=S.n_cells)
    return 1.0 / (len(np.unique(c)) * n[c])


def features(S, idx, kind, x0_mu=None, x0_sd=None):
    fp = S.fp(idx).astype(np.float64)
    if kind == "ecfp":
        return fp
    x = (S.x0[idx] - x0_mu) / x0_sd
    return np.hstack([fp, x])


def fit_all(S, st):
    """Returns a dict name -> predict(idx) closure plus per-alpha closures for the ridge models."""
    tr = S.train
    w = cell_weights(S, tr)
    x0_mu = S.x0[tr].mean(0); x0_sd = S.x0[tr].std(0) + 1e-6
    Y = S.delta(tr).astype(np.float64)
    M = st.cell_mean
    models = {}
    models["no_change"] = lambda idx: np.zeros((len(idx), 978), np.float32)
    models["train_cell_mean"] = lambda idx: M[S.cell_idx[idx]]
    Xa = features(S, tr, "ecfp_x0", x0_mu, x0_sd)
    ra = WRidge(Xa, Y, w)
    Xb = features(S, tr, "ecfp")
    rb = WRidge(Xb, Y - M[S.cell_idx[tr]], w)
    # alpha is on the unweighted-sum scale: weights sum to 1, so the penalty is alpha / n_train
    for al in ALPHAS:
        ba = ra.coef(al / len(tr)); bb = rb.coef(al / len(tr))
        models[f"ecfp_x0_ridge_a{al}"] = (lambda ba: lambda idx: ra.predict(features(S, idx, "ecfp_x0", x0_mu, x0_sd), ba).astype(np.float32))(ba)
        models[f"cellmean_ecfp_ridge_a{al}"] = (lambda bb: lambda idx: (M[S.cell_idx[idx]] + rb.predict(features(S, idx, "ecfp"), bb)).astype(np.float32))(bb)
    return models


def main():
    S = Store(); st = TrainStats(S)
    models = fit_all(S, st)
    val = S.val; yv = S.delta(val); uv = S.units.iloc[val].reset_index(drop=True)
    rows = []
    for name, f in models.items():
        p = f(val)
        m = unit_metrics(p, yv, uv.cell.to_numpy())
        s = summarise(m, uv, "source"); h = summarise(m, uv, "hcc_mean")
        rows.append(dict(model=name, val_rmse=s["rmse"], val_pearson=s["pearson"], val_centred_pearson=s["centred_pearson"], val_mrr=s["mrr"],
                         val_hcc_mean_rmse=h["rmse"]))
        print(rows[-1], flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(public("manifests", "response_geometry_r4", "baselines_validation.tsv"), sep="\t", index=False)
    sel = {}
    for fam in ["ecfp_x0_ridge", "cellmean_ecfp_ridge"]:
        t = tab[tab.model.str.startswith(fam + "_a")].sort_values(["val_rmse", "model"])
        sel[fam] = dict(model=t.model.iloc[0], alpha=int(t.model.iloc[0].split("_a")[-1]), val_rmse=float(t.val_rmse.iloc[0]))
    dump(dict(alpha_grid=ALPHAS, selection_rule="lowest validation source-panel delta RMSE", selected=sel),
         public("manifests", "response_geometry_r4", "baseline_selection.json"))
    print(json.dumps(sel, indent=1))


if __name__ == "__main__":
    main()
