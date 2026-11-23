"""Rank-gain metrics and paired lineage/compound resampling on matched score matrices."""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

AXES = ("within_drug_spearman", "within_drug_pearson", "ndcg_at_5", "precision_at_k",
        "within_cell_spearman")
RESIDUAL = ("residual_spearman", "residual_pearson", "residual_rmse")
EPS = 1e-12


def mean(x, axis=-1):
    valid = np.isfinite(x)
    count = valid.sum(axis=axis)
    total = np.where(valid, x, 0.).sum(axis=axis)
    return np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)


def association(y, p, *, axis, minimum=2, rank=False, constant=0.):
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = np.where(ok, y, np.nan), np.where(ok, p, np.nan)
    if rank:
        y, p = (rankdata(x, axis=axis, nan_policy="omit") for x in (y, p))
    count = ok.sum(axis=axis)
    yc, pc = (np.where(ok, x - np.expand_dims(mean(x, axis), axis), 0.) for x in (y, p))
    sy, sp = np.square(yc).sum(axis=axis), np.square(pc).sum(axis=axis)
    denominator = np.sqrt(sy * sp)
    score = np.divide((yc * pc).sum(axis=axis), denominator,
                      out=np.full(sy.shape, constant), where=(sy > EPS**2) & (sp > EPS**2))
    return np.where((count >= minimum) & (sy > EPS**2), score, np.nan)


def within_cell(y, p, k, precision_k):
    present = np.isfinite(y) & np.isfinite(p)
    n = present.sum(axis=-1)
    y, p = np.where(present, y, np.nan), np.where(present, p, np.nan)
    yr, pr = [rankdata(v, axis=-1, nan_policy="omit") for v in (y, p)]
    nonconstant = np.max(np.where(present, yr, -np.inf), axis=-1) > np.min(
        np.where(present, yr, np.inf), axis=-1)
    eligible = (n >= 2) & nonconstant
    # Rank gains and expected discounted gain within prediction ties match vc_build.evaluate.rank_metrics.
    gain = np.where(present, (yr - 1) / np.maximum(n[..., None] - 1, 1), 0.)
    k = min(k, y.shape[-1])
    d = 1 / np.log2(np.arange(2, k + 2))
    pred_order = np.argsort(-np.where(present, pr, -np.inf), axis=-1)[..., :k]
    ordered = np.take_along_axis(pr, pred_order, axis=-1)
    tied = ordered[..., :, None] == pr[..., None, :]
    ties = tied.sum(axis=-1)
    expected = np.divide(np.einsum("...ij,...j->...i", tied, gain), ties,
                         out=np.zeros(ties.shape), where=ties > 0)
    ideal = np.sort(gain, axis=-1)[..., -k:][..., ::-1] @ d
    ndcg = np.divide(expected @ d, ideal, out=np.zeros(ideal.shape), where=ideal > 0.)

    def top_membership(ranks):
        count = np.minimum(n, precision_k)
        cutoff = np.take_along_axis(np.sort(np.where(present, ranks, -np.inf), axis=-1),
                                    (ranks.shape[-1] - np.maximum(count, 1))[..., None], axis=-1)
        above, equal = (ranks > cutoff), (ranks == cutoff)
        fraction = (count - above.sum(axis=-1)) / np.maximum(1, equal.sum(axis=-1))
        return above + equal * fraction[..., None]

    overlap = np.sum(top_membership(yr) * top_membership(pr), axis=-1) / np.maximum(1, np.minimum(n, precision_k))
    spearman = association(yr, pr, axis=-1)
    return np.where(eligible, ndcg, np.nan), np.where(eligible, overlap, np.nan), spearman


def center(x):
    if not np.isfinite(x).all():
        raise ValueError("Double centering requires a complete patient-by-drug rectangle")
    out = x - x.mean(axis=-2, keepdims=True) - x.mean(axis=-1, keepdims=True) + x.mean(
        axis=(-2, -1), keepdims=True)
    return np.where(np.abs(out) < EPS, 0., out)


def metrics(y, p, *, precision_k=5, min_lineages=5, residuals=True):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if y.shape != p.shape or y.ndim not in (2, 3):
        raise ValueError("Expected aligned patient-by-drug matrices or batches of such matrices")
    drug = [association(y, p, axis=-2, minimum=min_lineages, rank=rank) for rank in (True, False)]
    by_cell = within_cell(y, p, 5, precision_k)
    result = {key: mean(v) for key, v in zip(AXES, [*drug, *by_cell])}
    if residuals:
        shape = (*y.shape[:-2], -1)
        yy, pp = (center(m).reshape(shape) for m in (y, p))
        result.update({
            "residual_spearman": association(yy, pp, axis=-1, rank=True),
            "residual_pearson": association(yy, pp, axis=-1),
            "residual_rmse": np.sqrt(np.mean(np.square(yy - pp), axis=-1)),
        })
    return {key: float(value) for key, value in result.items()} if y.ndim == 2 else result


def bootstrap(y, forecasts, *, repeats=2400, seed=11942):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    cells, drugs = y.shape
    results = {model: [] for model in forecasts}
    for start in range(0, repeats, 50):
        size = min(50, repeats - start)
        people = rng.integers(cells, size=(size, cells, 1))
        compounds = rng.integers(drugs, size=(size, 1, drugs))
        obs = y[people, compounds]
        for model, p in forecasts.items():
            results[model].append(pd.DataFrame(metrics(obs, p[people, compounds])))
    return {name: pd.concat(values, ignore_index=True) for name, values in results.items()}


def interval(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return tuple(map(float, np.quantile(finite, [.025, .975]))) if len(finite) else (np.nan, np.nan)


def pivot(frame, value, *, row="lineage_group"):
    return frame.pivot(index=row, columns="compound_id", values=value).sort_index().sort_index(axis=1)


def top5(p):
    # Columns are sorted by InChIKey before this call. Stable score ties therefore
    # have one deterministic representative only for set overlap and diversity.
    return [set(row) for row in np.argsort(-p, axis=1, kind="stable")[:, :5]]
