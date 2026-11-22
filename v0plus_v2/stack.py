"""Learn nonnegative residual mixtures with global and drug-specific shrinkage."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import nnls

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

    def predict(self, arms, anchor, drug):
        x = design(arms, anchor, self.columns)
        weights = np.stack([self.group_weights.get(group, self.global_weights) for group in drug])
        return anchor + self.amplitude*(x*weights).sum(axis=1)


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
