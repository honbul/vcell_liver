"""Common-row scoring with global-lineage paired resampling across held screens."""

import numpy as np
import pandas as pd

from vc_build.response_components import orthogonal_parts

from .core_reference_metrics import AXES, association, mean, within_cell

RESIDUAL = "residual_spearman"
CORE_METRICS = ("ndcg_at_5", "within_drug_spearman", "precision_at_k")
ALL_METRICS = (*AXES, RESIDUAL)


def matrix(group, drugs, values):
    collapsed = group.groupby(["lineage_group", "compound_id"], sort=True)[values].mean()
    return {key: collapsed[key].unstack().reindex(columns=drugs) for key in values}


def blocks(frame, models):
    values = ["observed", *models]
    for name, group in frame.groupby(["source_id", "endpoint_id", "fold"], sort=True):
        mats = matrix(group, sorted(group.compound_id.unique()), values)
        truth = mats["observed"]
        obs = truth.to_numpy()
        preds = {name: mats[name].to_numpy() for name in models}
        if any(not np.array_equal(np.isfinite(obs), np.isfinite(p)) for p in preds.values()):
            raise ValueError("Model and target support differ after lineage aggregation")
        yield name, truth.index, obs, preds


def residual_association(y, p):
    mask = np.isfinite(y)
    parts = []
    for values in (y, p):
        component = orthogonal_parts(values, mask.astype(float))[2][mask]
        # Exact additive surfaces can leave roundoff at the eigensolver scale.
        tolerance = 1e-12 * max(1., np.abs(values[mask]).max())
        parts.append(np.where(np.abs(component) <= tolerance, 0., component))
    return float(association(*parts, axis=-1, minimum=5, rank=True))


class WeightedRanking:
    """Average tied ranks of literal lineage multiplicities without expanding repeated records."""

    def __init__(self, vector):
        self.order = np.argsort(vector, kind="stable")
        self.inverse = np.argsort(self.order)
        sorted_v = vector[self.order]
        self.starts = np.r_[0, np.flatnonzero(sorted_v[1:] != sorted_v[:-1]) + 1]
        self.groups = np.cumsum(np.r_[True, sorted_v[1:] != sorted_v[:-1]]) - 1

    def ranks(self, counts):
        tied = np.add.reduceat(counts[:, self.order], self.starts, axis=1)
        ascending = np.cumsum(tied, axis=1) - .5 * tied + .5
        return ascending[:, self.groups][:, self.inverse]


def weighted_association(y, p, counts, minimum=5):
    n = counts.sum(axis=1).clip(min=1)
    yy = y - (y * counts).sum(axis=1)[:, None] / n[:, None]
    pp = p - (p * counts).sum(axis=1)[:, None] / n[:, None]
    sy, sp = (counts * yy**2).sum(axis=1), (counts * pp**2).sum(axis=1)
    covariance = (counts * yy * pp).sum(axis=1)
    corr = np.divide(covariance, np.sqrt(sy * sp), out=np.zeros(len(sy)), where=(sy > 1e-24) & (sp > 1e-24))
    ok = (n >= minimum) & (sy > 1e-24)
    return np.where(ok, corr, np.nan)


def screen_scores(frame, models, reweights=None, *, minimum=5):
    """Use equal lineages for cell metrics and lineage-weighted mean across disjoint folds."""
    result = {}
    for (screen, _, _), lineages, y, p in blocks(frame, models):
        store = result.setdefault(screen, {name: [] for name in models})
        weights = None if reweights is None else reweights[1][:, reweights[0].get_indexer(lineages)]
        for model, prediction in p.items():
            c = within_cell(y, prediction, 5, 5)
            by_drug = [association(y, prediction, axis=0, minimum=minimum, rank=rank)
                       for rank in (True, False)]
            point = dict(zip(AXES, (float(mean(v)) for v in (*by_drug, *c))))
            point[RESIDUAL] = residual_association(y, prediction)
            point_weights = {axis: len(lineages) for axis in ALL_METRICS}
            for axis, per_line in zip(("ndcg_at_5", "precision_at_k", "within_cell_spearman"), c):
                point_weights[axis] = np.isfinite(per_line).sum()
            boot = None
            if weights is not None:
                boot = {}
                for key, per_line in zip(("ndcg_at_5", "precision_at_k"), c[:2]):
                    available = np.isfinite(per_line)
                    boot[key] = (weights[:, available] @ per_line[available], weights[:, available].sum(axis=1))
                spearmans = []
                for drug in range(y.shape[1]):
                    idx = np.isfinite(y[:, drug])
                    if idx.sum() < minimum:
                        continue
                    w = weights[:, idx]
                    yr = WeightedRanking(y[idx, drug]).ranks(w)
                    pr = WeightedRanking(prediction[idx, drug]).ranks(w)
                    spearmans.append(weighted_association(yr, pr, w, minimum))
                drug_average = mean(np.asarray(spearmans).T)
                count = np.where(np.isfinite(drug_average), weights.sum(axis=1), 0)
                boot["within_drug_spearman"] = (np.nan_to_num(drug_average) * count, count)
            store[model].append((point_weights, point, boot))

    def aggregate(block_values):
        ns, points, resamples = zip(*block_values)
        point = {}
        for axis in ALL_METRICS:
            val = np.array([r[axis] for r in points])
            weight = np.array([w[axis] for w in ns])
            ok = np.isfinite(val) & (weight > 0)
            point[axis] = np.average(val[ok], weights=weight[ok]) if ok.any() else np.nan
        boot = {}
        if reweights is not None:
            for axis in CORE_METRICS:
                numer, denom = np.sum([r[axis] for r in resamples], axis=0)
                boot[axis] = numer / denom.clip(min=1)
        return point, boot

    return {screen: {model: aggregate(parts) for model, parts in shelf.items()}
            for screen, shelf in result.items()}


def shared_draws(frame, count, seed):
    lineages = pd.Index(sorted(frame.lineage_group.unique()))
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(lineages), np.ones(len(lineages)) / len(lineages), size=count)
    return lineages, weights
