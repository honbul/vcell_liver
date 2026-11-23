"""Batch scoring with the project's tie-aware rank relevance and fixed observed mask."""
import numpy as np
from scipy.stats import rankdata

from vc_build.v0plus_score import correlations

NAMES = ("ndcg_at_5", "precision_at_5", "within_sample_spearman", "within_drug_spearman",
         "within_drug_pearson", "interaction_pearson", "interaction_spearman", "mae", "rmse")
DISCOUNT = 1. / np.log2(np.arange(2, 7))


def ranks(values, axis):
    return rankdata(values, axis=axis, nan_policy="omit")


def row_corr(a, b):
    return correlations(a[:, :, None], b[:, :, None])[:, 0] if a.shape[1] >= 5 else np.full(len(a), np.nan)


class Retrieval:
    """Vectorized rank-gain NDCG and expected top-5 overlap."""
    def __init__(self, y):
        self.mask = np.isfinite(y)
        self.rank = ranks(y, axis=1)
        count = self.mask.sum(axis=1)
        self.valid = (count >= 5) & (np.nanmax(y, axis=1) > np.nanmin(y, axis=1))
        self.gain = (self.rank - 1) / np.maximum(count[:, None] - 1, 1)
        self.ideal = np.sort(np.nan_to_num(self.gain, nan=-1.), axis=1)[:, -5:][:, ::-1] @ DISCOUNT
        self.truth_top = self.top(np.where(self.mask, y, -np.inf))

    @staticmethod
    def top(values):
        cutoff = np.partition(values, -5, axis=1)[:, -5:]
        at_five = cutoff.min(axis=1, keepdims=True)
        exceed = values > at_five
        tie = values == at_five
        return exceed + tie * (5 - exceed.sum(axis=1, keepdims=True)) / tie.sum(axis=1, keepdims=True)

    def __call__(self, forecasts, full=False):
        p = np.where(self.mask, forecasts, -np.inf)
        order = np.argpartition(-p, 4, axis=1)[:, :5]
        values = np.take_along_axis(p, order, axis=1)
        values = np.sort(values, axis=1)[:, ::-1]
        ties = (p[:, None, :] == values[:, :, None]) & self.mask[:, None, :]
        ties_count = np.maximum(ties.sum(axis=2), 1)
        gains = (ties * np.nan_to_num(self.gain)[:, None, :]).sum(axis=2) / ties_count
        ndcg = (gains @ DISCOUNT) / np.maximum(self.ideal, 1e-9)
        precision = (self.top(p) * self.truth_top).sum(axis=1) / 5
        terms = [ndcg, precision]
        if full:
            terms.append(row_corr(self.rank, ranks(np.where(self.mask, p, np.nan), 1)))
        return np.where(self.valid[:, None], np.column_stack(terms), np.nan)


def center2(values):
    """Two-way residuals on observed entries by alternating projections."""
    mask = np.isfinite(values)
    result = np.where(mask, values, 0.).astype(float, copy=False)
    rows = np.maximum(mask.sum(axis=-1, keepdims=True), 1)
    cols = np.maximum(mask.sum(axis=-2, keepdims=True), 1)
    # Use each draw's magnitude to identify a degenerate interaction after projection.
    tolerance = 64 * np.finfo(result.dtype).eps * np.max(np.abs(result), axis=(-2, -1), keepdims=True)
    for _ in range(24):
        result -= mask * (result.sum(axis=-1, keepdims=True) / rows)
        change = result.sum(axis=-2, keepdims=True) / cols
        result -= mask * change
        if np.all(np.max(np.abs(change), axis=(-2, -1), keepdims=True) <= tolerance):
            break
    degenerate = np.max(np.abs(result), axis=(-2, -1), keepdims=True) <= 8 * tolerance
    return np.where(mask, np.where(degenerate, 0., result), np.nan)


def mean_valid(a, axis):
    valid = np.isfinite(a)
    count = valid.sum(axis=axis)
    return np.divide(np.nansum(a, axis=axis), count, out=np.full(count.shape, np.nan), where=count > 0)


def sample_summaries(actual, pred):
    retrieval = Retrieval(actual)(pred, full=True)
    e = pred - actual
    return np.column_stack((retrieval, mean_valid(np.abs(e), 1), mean_valid(e * e, 1)))


def batch_stats(y, p, sample, *, absolute=True):
    """One row per bootstrap draw; lineages stay within their original fold."""
    if y.ndim != 3 or y.shape != p.shape:
        raise ValueError("Expected draws × independent groups × compounds")
    have = np.isfinite(y)
    p = np.where(have, p, np.nan)
    sp = mean_valid(correlations(ranks(y, 1), ranks(p, 1)), 1)
    pcc = mean_valid(correlations(y, p), 1)
    a = center2(y).reshape(len(y), -1, 1)
    b = center2(p).reshape(len(p), -1, 1)
    ixp = correlations(a, b)[:, 0]
    ixs = correlations(ranks(a, 1), ranks(b, 1))[:, 0]
    sample_metrics = mean_valid(sample, 1)
    mae = sample_metrics[:, 3] if absolute else np.full(len(y), np.nan)
    rmse = np.sqrt(sample_metrics[:, 4]) if absolute else np.full(len(y), np.nan)
    return np.column_stack((sample_metrics[:, :3], sp, pcc, ixp, ixs, mae, rmse))
