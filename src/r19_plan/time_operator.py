"""Train a small time mapping on exact, eligible LINCS source pairs only."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.r19_plan.dev_selection import checked_array

PAIR_KEY = ["study", "cell_id", "parent", "dose_uM"]
RANKS = (4, 16, 32)
RIDGES = (.1, 1.)


@dataclass(frozen=True)
class TimeOperator:
    genes: tuple[int, ...]
    input_mean: np.ndarray
    output_mean: np.ndarray
    weights: np.ndarray
    rank: int
    ridge: float
    fit_parent_count: int

    def predict(self, source, genes):
        if tuple(genes) != self.genes:
            raise ValueError("The time operator gene order changed")
        prediction_input = checked_array(source, (len(source), len(self.genes)), "time-operator source query")
        return self.output_mean + (prediction_input - self.input_mean) @ self.weights


def fit_time_operator(rows, earlier, later, genes, *, rank, ridge, held_parents):
    """Fit 6 h-to-24 h mapping on source parent-balanced exact condition pairs."""
    if not isinstance(rows, pd.DataFrame) or not len(rows) or not set(PAIR_KEY) <= set(rows):
        raise ValueError("Expected registered exact-pair source metadata")
    if rows[PAIR_KEY].isna().any().any() or rows[PAIR_KEY].astype(str).eq("").any().any():
        raise ValueError("A time-paired source condition is missing")
    dose = pd.to_numeric(rows.dose_uM, errors="raise")
    if rows.assign(dose_uM=dose).duplicated(PAIR_KEY).any() or not np.isfinite(dose).all() or not dose.gt(0).all():
        raise ValueError("Every time-paired source condition needs a unique measured dose")
    if set(rows.parent) & set(held_parents):
        raise ValueError("A held parent entered the time operator's source pairs")
    if rank not in RANKS or ridge not in RIDGES:
        raise ValueError("Time operator setting is outside the registered rank and ridge grid")
    genes = tuple(genes)
    if not genes or len(genes) != len(set(genes)):
        raise ValueError("The time operator requires a unique, nonempty landmark axis")
    earlier = checked_array(earlier, (len(rows), len(genes)), "6 h source response")
    later = checked_array(later, earlier.shape, "24 h source response")
    parent_count = rows.parent.nunique()
    if parent_count <= rank:
        raise ValueError(f"Insufficient independent source parents for time operator rank {rank}")

    # Each canonical parent contributes the same total weight across exact conditions.
    weight = 1. / rows.groupby("parent").parent.transform("size").to_numpy()
    input_mean = np.average(earlier, axis=0, weights=weight)
    output_mean = np.average(later, axis=0, weights=weight)
    x = (earlier - input_mean) * np.sqrt(weight)[:, None]
    y = (later - output_mean) * np.sqrt(weight)[:, None]
    left, values, right = np.linalg.svd(x, full_matrices=False)
    tolerance = values[0] * np.finfo(x.dtype).eps * max(x.shape)
    if np.count_nonzero(values > tolerance) < rank:
        raise ValueError(f"Insufficient numerical support for time operator rank {rank}")
    shrink = values[:rank] / (np.square(values[:rank]) + ridge)
    weights = (right[:rank].T * shrink) @ (left[:, :rank].T @ y)
    return TimeOperator(genes, input_mean, output_mean, weights, rank, ridge, parent_count)
