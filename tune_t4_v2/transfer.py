"""Centered zero-shot transfer and nested PHH adaptations with scaffold budgets."""

import numpy as np
from sklearn.linear_model import Ridge

from .data import parent_weights
from .linear import conditions

BUDGETS = (5, 10, 20)


def centered_prior(model, reference_x, target_rows, x, *, device="cpu"):
    center = np.mean(reference_x, axis=0)
    shifted = center + (model.y_mean - model.x_mean)
    # DMSO-centered signatures have a zero-effect origin; preserve the source time displacement.
    return model.predict(target_rows, x, input_mean=center, output_anchor=shifted, device=device)


def fewshot(fit, k, seed):
    rng = np.random.default_rng(seed)
    registry = fit.rows[["scaffold", "parent"]].drop_duplicates()
    scaffold_ids = np.asarray(sorted(registry.scaffold.unique()))
    if len(scaffold_ids) < k:
        return None
    selected = rng.choice(scaffold_ids, size=k, replace=False)
    representatives = set()
    for scaffold in selected:
        drugs = sorted(registry.loc[registry.scaffold.eq(scaffold), "parent"])
        representatives.add(rng.choice(drugs))
    return fit.select(fit.rows.parent.isin(representatives).to_numpy())


def adapt(predicted, target_rows, target_x, fit, prior_fit, alpha, rank):
    weight = parent_weights(fit.rows)
    residuals = fit.y - prior_fit
    center = np.average(residuals, axis=0, weights=weight)
    if rank == 0:
        return predicted + center
    standardized = fit.x - fit.x.mean(axis=0)
    _, _, components = np.linalg.svd(standardized, full_matrices=False)
    basis = components[:min(rank, len(components)-1)]
    c = conditions(fit.rows, fit.x, False)[:, :2]
    t = conditions(target_rows, target_x, False)[:, :2]
    train_features = np.c_[standardized @ basis.T, c]
    test_features = np.c_[(target_x-fit.x.mean(axis=0)) @ basis.T, t]
    model = Ridge(alpha=alpha).fit(train_features, residuals, sample_weight=weight)
    return predicted + model.predict(test_features)
