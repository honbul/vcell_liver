"""Summarize operational I-track gains on frozen complete-view support."""

import numpy as np
import pandas as pd

from src.r19_plan.i_score import (
    BASELINES, BOOTSTRAPS, FOLDS, MIN_FOLD_PARENTS, MIN_PARENTS, checked_array,
    checked_rows, coverage_arrays, parent_summaries, squared_rows,
)

PRIMARY = ("I1", "I2", "I3", "I4")
SUPPLEMENTAL = ("I3_all",)


def relative_interval(rmse):
    """Use the paired reference RMSE in every parent-cluster bootstrap draw."""
    n = len(rmse)
    if n < 2:
        return np.nan, np.nan, 0

    rng = np.random.default_rng(150104)
    draws = np.empty(BOOTSTRAPS)
    reference = rmse[:, 1]
    improvement = reference - rmse[:, 0]
    for start in range(0, BOOTSTRAPS, 128):
        stop = min(start + 128, BOOTSTRAPS)
        sample = rng.integers(n, size=(stop - start, n))
        denominator = reference[sample].sum(axis=1)
        numerator = improvement[sample].sum(axis=1)
        draws[start:stop] = np.divide(numerator, denominator, out=np.full(stop - start, np.nan),
                                      where=denominator > 0)
    valid = int(np.isfinite(draws).sum())
    if valid != BOOTSTRAPS:
        return np.nan, np.nan, valid
    low, high = np.quantile(draws, [0.005, 0.995])
    return low, high, valid


def comparison_row(rows, truth_genes, energy, arm_error, reference_error, *, pooled):
    rmse, sse, exposure = parent_summaries(rows, truth_genes, energy, arm_error, reference_error)
    n = len(rmse)
    candidate, reference = rmse.mean(axis=0) if n else (np.nan, np.nan)
    relative = (reference - candidate) / reference if reference > 0 else np.nan
    total = exposure.sum()
    r2_delta = (sse[:, 1] - sse[:, 0]).sum() / total if total > 0 else np.nan
    low, high, valid = relative_interval(rmse) if pooled else (np.nan, np.nan, 0)
    fold_count = rows.groupby("fold").parent.nunique().reindex(FOLDS, fill_value=0)
    eligible = (n >= MIN_PARENTS and fold_count.ge(MIN_FOLD_PARENTS).all()
                if pooled else n >= MIN_FOLD_PARENTS)
    return dict(parents=n, rows=len(rows), eligible=bool(eligible),
                candidate_rmse=candidate, reference_rmse=reference, relative_rmse=relative,
                relative_rmse_low99=low, relative_rmse_high99=high, valid_bootstrap_draws=valid,
                delta_r2=r2_delta, rmse_improves=bool(relative > 0), r2_improves=bool(r2_delta > 0),
                jointly_improves=bool(relative > 0 and r2_delta > 0))


def operational_points(rows, truth, forecasts, coverage):
    """Return pooled and fold-level effects for I1–I4 and the three-view arm.

    The caller supplies verified frozen arrays on the score-row axis. The same
    complete-view rows enter both comparisons for each arm. Positive gains
    favor the candidate. Relative RMSE and R² are reported as fractions.
    """
    rows = checked_rows(rows, fit_parents_by_fold=None)
    truth = np.asarray(truth)
    if truth.ndim != 2 or truth.shape[0] != len(rows) or truth.shape[1] == 0:
        raise ValueError("I operational scoring expects a well-by-gene score target")
    truth = checked_array(truth, truth.shape, "score target")
    candidates = (*PRIMARY, *SUPPLEMENTAL)
    if not {*candidates, *BASELINES} <= set(forecasts):
        raise ValueError("I operational scoring requires the registered candidates and reference arms")
    totals, counts = coverage_arrays(rows, coverage, forecasts)
    energy = squared_rows(truth)
    references = {arm: squared_rows(truth, checked_array(forecasts[arm], truth.shape, arm))
                  for arm in BASELINES}
    effects = []
    for candidate in candidates:
        complete = counts[candidate] == totals[candidate]
        error = squared_rows(truth, checked_array(forecasts[candidate], truth.shape, candidate))
        for reference, ref_error in references.items():
            group = "primary" if candidate in PRIMARY else "supplemental"
            for fold in ("all", *FOLDS):
                pooled = fold == "all"
                keep = complete if pooled else complete & rows.fold.eq(fold).to_numpy()
                effects.append(dict(arm=candidate, family=group, reference=reference, fold=fold,
                                    scope="complete_shared", **comparison_row(
                                        rows.loc[keep], truth.shape[1], energy[keep], error[keep],
                                        ref_error[keep], pooled=pooled)))
    return pd.DataFrame(effects)
