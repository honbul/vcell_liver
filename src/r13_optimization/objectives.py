"""R12-compatible macro RMSE, with precomputed quadratic scores for residual gains."""

import numpy as np

from src.r11_chemical_probe.metrics import Contrast


def center(metric, values):
    out = np.zeros_like(values, dtype=np.float64)
    for ix, weight in zip(metric.pools, metric.centers):
        pool = values[ix]
        out[ix] = pool - np.average(pool, axis=0, weights=weight)
    return out


def row_products(a, b):
    return np.einsum("ig,ig->i", a, b, dtype=np.float64) / a.shape[1]


class GainScores:
    def __init__(self, rows, truth, base, minimum):
        self.metric = Contrast(rows, minimum)
        self.error = np.asarray(base, dtype=np.float64) - truth
        self.centered = center(self.metric, self.error)
        self.square = row_products(self.error, self.error)
        self.centered_square = row_products(self.centered, self.centered)
        self.truth = truth

    def points(self, adjustment, gains):
        if adjustment.shape != self.error.shape or not np.isfinite(adjustment).all():
            raise ValueError("Invalid query adjustment shape or values")
        centered_adjustment = center(self.metric, adjustment)
        abs_mse = self.square + 2 * gains[:, None] * row_products(self.error, adjustment)
        abs_mse += gains[:, None] ** 2 * row_products(adjustment, adjustment)
        ctr_mse = self.centered_square + 2 * gains[:, None] * row_products(self.centered, centered_adjustment)
        ctr_mse += gains[:, None] ** 2 * row_products(centered_adjustment, centered_adjustment)
        absolute = np.average(np.maximum(abs_mse, 0) ** 0.5, axis=1, weights=self.metric.all_weight)
        contrast = np.average(np.maximum(ctr_mse, 0) ** 0.5, axis=1, weights=self.metric.primary_weight)
        return contrast, absolute

    def diagnostics(self, adjustment):
        predicted = center(self.metric, adjustment)
        measured = center(self.metric, self.truth)
        cross = row_products(predicted, measured)
        norm = np.sqrt(row_products(predicted, predicted) * row_products(measured, measured))
        valid = (norm > 1e-15) & (self.metric.primary_weight > 0)
        if not np.any(valid):
            return 0.0, None
        w = self.metric.primary_weight[valid]
        covered = w.sum() / self.metric.primary_weight.sum()
        return float(covered), float(np.average(cross[valid] / norm[valid], weights=w))
