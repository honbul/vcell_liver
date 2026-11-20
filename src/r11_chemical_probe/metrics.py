"""Compound-balanced evaluation with exact positive-weight re-centering."""

import numpy as np

from src.r6_followup_v1.metrics import cond_rmse, grams

from .dataset import ENTITY, GROUP, KEY, POOL, group_weights, weights_for


class Contrast:
    def __init__(self, rows, minimum):
        self.rows = rows.reset_index(drop=True)
        eligible = self.rows.groupby(POOL, sort=False)[GROUP].transform("nunique").ge(minimum)
        self.active = np.flatnonzero(eligible)
        if not len(self.active):
            raise ValueError("No metadata-eligible contrast pools")
        self.pools = list(self.rows[eligible].groupby(POOL, sort=False).groups.values())
        self.pools = [idx.to_numpy() for idx in self.pools]
        self.group_ids, self.inverse = np.unique(self.rows[GROUP], return_inverse=True)
        self.centers = [
            weights_for(self.rows.loc[idx], (GROUP, ENTITY, KEY[0])) for idx in self.pools
        ]
        self.all_weight = group_weights(self.rows)
        self.primary_weight = np.zeros(len(self.rows))
        self.primary_weight[self.active] = group_weights(self.rows.loc[self.active])

    def point(self, predicted, observed):
        return self.points(predicted, observed)[0]

    def _shape(self, predicted, observed):
        if predicted.shape != observed.shape or len(predicted) != len(self.rows):
            raise ValueError("Prediction/observation row or gene dimensions differ")
        if not np.isfinite(predicted).all() or not np.isfinite(observed).all():
            raise ValueError("Nonfinite response vectors")

    def points(self, predicted, observed):
        self._shape(predicted, observed)
        error = predicted.astype(np.float64) - observed
        values = np.full(len(self.rows), np.nan)
        for idx, weight in zip(self.pools, self.centers):
            values[idx] = np.sqrt(np.mean((error[idx] - np.average(error[idx], axis=0, weights=weight)) ** 2, axis=1))
        primary = np.average(values[self.active], weights=self.primary_weight[self.active])
        delta = cond_rmse(predicted, observed)
        return float(primary), float(np.average(delta, weights=self.all_weight)), values, delta

    def bootstrap_values(self, predicted, observed, group_draws):
        """Recompute each pool's mean under every draw, using the R6 Gram identity."""
        self._shape(predicted, observed)
        matrix = grams(predicted.astype(np.float64) - observed, self.pools)
        result = np.full((len(group_draws), len(self.rows)), np.nan)
        for kernel, idx, center_weight in zip(matrix, self.pools, self.centers):
            weight = group_draws[:, self.inverse[idx]] * center_weight
            weight /= weight.sum(axis=1, keepdims=True)
            kernel_mean = weight @ kernel
            variance = np.diag(kernel) - 2 * kernel_mean + (kernel_mean * weight).sum(axis=1, keepdims=True)
            result[:, idx] = np.sqrt(np.clip(variance, 0, None))
        return result

    def retrieval(self, predicted, observed):
        """Tie-aware percentile rank and expected top-1 within pool, by entity."""
        self._shape(predicted, observed)
        rank = np.full(len(self.rows), np.nan)
        top = rank.copy()
        n = np.zeros(len(self.rows), dtype=np.int64)
        same_scores = np.zeros(len(self.rows), dtype=bool)
        for idx, center_weight in zip(self.pools, self.centers):
            rows = self.rows.loc[idx]
            entities, inverse = np.unique(rows[ENTITY], return_inverse=True)
            pred = predicted[idx].astype(np.float64)
            truth = observed[idx].astype(np.float64)
            pred -= np.average(pred, axis=0, weights=center_weight)
            truth -= np.average(truth, axis=0, weights=center_weight)
            p = np.stack([pred[inverse == i].mean(axis=0) for i in range(len(entities))])
            t = np.stack([truth[inverse == i].mean(axis=0) for i in range(len(entities))])
            p_norm = np.linalg.norm(p, axis=1, keepdims=True)
            t_norm = np.linalg.norm(t, axis=1, keepdims=True)
            score = (p / np.maximum(p_norm, 1e-12)) @ (t / np.maximum(t_norm, 1e-12)).T
            score[p_norm.ravel() <= 1e-12] = 0
            correct = score.diagonal()[:, None]
            ties = np.isclose(score, correct, atol=1e-10, rtol=1e-8)
            better = ((score > correct) & ~ties).sum(axis=1)
            n_tie = ties.sum(axis=1)
            rank[idx] = (1 - (better + (n_tie - 1) / 2) / (len(entities) - 1))[inverse]
            top[idx] = (np.where(better == 0, 1 / n_tie, 0))[inverse]
            n[idx] = len(entities)
            same_scores[idx] = (p_norm.ravel() <= 1e-12)[inverse]
        return rank, top, n, same_scores


def correlations(predicted, observed, pearson=False):
    a = predicted.astype(np.float64)
    b = observed.astype(np.float64)
    if pearson:
        a -= a.mean(axis=1, keepdims=True)
        b -= b.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    values = np.full(len(a), np.nan)
    valid = denom > 1e-12
    values[valid] = np.einsum("ij,ij->i", a[valid], b[valid]) / denom[valid]
    reason = "NA_zero_variance" if pearson else "NA_zero_norm"
    return values, np.where(valid, "", reason)
