"""Finite-sample calibration for a declared exchangeable evaluation population."""

from __future__ import annotations

from math import ceil

import numpy as np


def absolute_error_quantile(predicted, observed, coverage: float = 0.90) -> float:
    if not 0 < coverage < 1:
        raise ValueError("Coverage must be strictly between zero and one")
    predicted, observed = np.asarray(predicted), np.asarray(observed)
    if predicted.ndim != 1 or predicted.shape != observed.shape:
        raise ValueError("Calibration arrays must be aligned vectors")
    residual = np.abs(predicted - observed)
    if not np.isfinite(residual).all():
        raise ValueError("Calibration observations must be finite")
    return finite_sample_quantile(residual, coverage)


def finite_sample_quantile(scores, coverage: float) -> float:
    if not 0 < coverage < 1:
        raise ValueError("Coverage must be strictly between zero and one")
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Expected nonnegative, finite calibration scores")
    index = ceil((len(scores) + 1) * coverage) - 1
    if index >= len(scores):
        raise ValueError("Insufficient independent calibration units for requested coverage")
    return float(np.partition(scores, index)[index])
