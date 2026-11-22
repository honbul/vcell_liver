"""Learn nonnegative residual mixtures with global and drug-specific shrinkage."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from scipy.special import expit

LINEAR = ("adaptive", "ridge", "rbf", "cosine", "enet", "pls")
TREES = ("drug_lgbm", "drug_xgboost", "drug_catboost")
NEURAL = ("bilinear_rna", "two_tower_rna", "two_tower_fm", "hypernet_rna", "hypernet_fm",
          "bilinear_fm", "two_tower_cmap", "two_tower_weight", "two_tower_smooth_only")
RECIPES = {
    "balanced_global": (("rbf", "ridge", *TREES, *NEURAL), 16., None),
    "balanced_drug": (("rbf", "ridge", *TREES, *NEURAL), 16., 20.),
    "adaptive_mix": (("adaptive", "rbf", "cosine", *TREES, *NEURAL), 12., 12.),
    "linear_ranking": (("ridge", "rbf", "cosine", "enet"), 16., 30.),
    "nonlinear_borrowing": (("adaptive", *TREES, *NEURAL), 15., 35.),
    "pairwise_rank": (("rbf", "ridge", *TREES, *NEURAL), 16., 20.),
    "pairwise_top5": (("adaptive", "rbf", "cosine", *TREES, *NEURAL), 12., 12.),
    "foundation_agnostic": (("rbf", "ridge", *TREES, "bilinear_rna", "two_tower_rna", "hypernet_rna"), 16., 12.),
}
CAPS = (.45, .65, .85, 1., 1.2, 1.45)


def design(arms, anchored, columns):
    data = arms.loc[:, columns].to_numpy(dtype=float)
    residual = data-anchored[:, None]
    # A family with insufficient support falls back to the anchor for that drug.
    return np.nan_to_num(residual, nan=0., posinf=0., neginf=0.)


@dataclass
class MetaHead:
    columns: list
    global_weights: np.ndarray
    group_weights: dict
    recipe: str
    amplitude: float = 1.
    offsets: dict = field(default_factory=dict)

    def predict(self, arms, anchor, drug):
        x = design(arms, anchor, self.columns)
        weights = np.stack([self.group_weights.get(group, self.global_weights) for group in drug])
        shifts = drug.map(self.offsets).fillna(0.).to_numpy()
        return anchor + shifts + self.amplitude*(x*weights).sum(axis=1)


def solve(x, y, prior, penalty):
    reward = np.sqrt(penalty)
    m = np.vstack([x, reward*np.eye(x.shape[1])])
    t = np.r_[y, reward*prior]
    return nnls(m, t)[0]


def build(arms, targets, recipe):
    names, lam, by_drug = RECIPES[recipe]
    cols = [name for name in names if name in arms]
    if not cols:
        raise ValueError("No submitted arms for the selected stack")
    anchors = targets.anchor.to_numpy()
    drug = targets.compound_id.to_numpy()
    values = targets.residual.to_numpy()
    p = design(arms, anchors, cols)
    usable = np.isfinite(anchors) & np.isfinite(values)
    if usable.sum() < 200:
        raise ValueError("Insufficient measured OOF residuals for stacking")
    mass = np.maximum(np.median(np.square(p[usable]).sum(axis=1)), .001)
    prior = np.asarray([.75/sum(n in LINEAR for n in cols) if k in LINEAR else
                        .25/max(1, sum(n not in LINEAR for n in cols)) for k in cols])
    weights = solve(p[usable], values[usable], prior, lam*mass)
    per_drug = {}
    if by_drug is not None:
        for compound in np.unique(drug):
            mask = usable & (compound==drug)
            if mask.sum() >= 16:
                per_drug[compound] = solve(p[mask], values[mask], weights, by_drug*mass)
    return MetaHead(cols, weights, per_drug, recipe)


def pivot(frame):
    return frame.pivot(index=["inner", "cell_id", "compound_id"], columns="method", values="forecast")

def fit_offsets(rows, drug_ids, focus_top, shrink):
    """Calibrate the ranking anchor on paired native within-cell comparisons."""
    anchor_margin, direction, weights, first, second = [], [], [], [], []
    index = pd.Index(drug_ids)
    for _, block in rows.groupby(["source_id","endpoint_id","cell_id","inner"],sort=False):
        n = len(block)
        if n<3:
            continue
        a,b = np.triu_indices(n,1)
        observed = block.observed.to_numpy()
        response = np.sign(observed[a]-observed[b])
        present = response!=0
        if not present.any():
            continue
        scores = block.predicted_value.to_numpy()
        ids = index.get_indexer(block.compound_id)
        priority = np.zeros(n,dtype=bool)
        priority[np.argsort(-observed,kind="stable")[:min(5,n)]]=True
        importance = 1. + 2.*focus_top*(priority[a] | priority[b])
        anchor_margin.append((scores[a]-scores[b])[present])
        direction.append(response[present])
        weights.append((importance/np.sum(importance[present]))[present])
        first.append(ids[a[present]])
        second.append(ids[b[present]])
    if not anchor_margin:
        return {}
    margin, y, mass, a, b = map(np.concatenate,(anchor_margin,direction,weights,first,second))
    mass /= np.sum(mass)
    scale = .105

    def objective(offsets):
        logits = y*(margin+offsets[a]-offsets[b])/scale
        values = np.logaddexp(0.,-logits)
        e = -y*expit(-logits)*mass/scale
        grad = np.bincount(a,weights=e,minlength=len(index))-np.bincount(b,weights=e,minlength=len(index))
        penalty = shrink/(len(index)*scale**2)
        return values@mass + .5*penalty*(offsets@offsets), grad+penalty*offsets

    f = minimize(objective,np.zeros(len(index)),jac=True,method="L-BFGS-B",
                 bounds=[(-.11,.11)]*len(index),options={"maxiter":65,"ftol":1e-8})
    if not f.success:
        raise ValueError(f"Pairwise rank anchor did not converge: {f.message}")
    return dict(zip(index, f.x-f.x.mean()))
